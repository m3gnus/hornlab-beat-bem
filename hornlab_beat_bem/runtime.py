"""Julia discovery and truthful BEAT backend capability probing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import BEAT_CPU, BEAT_CUDA, BEAT_METAL, BEAT_ROCM

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SOLVER_SCRIPT = PACKAGE_DIR / "julia" / "solver.jl"
CPU_PROJECT = PACKAGE_DIR / "julia"
CUDA_PROJECT = PACKAGE_DIR / "julia_cuda"
ROCM_PROJECT = PACKAGE_DIR / "julia_rocm"
METAL_PROJECT = PACKAGE_DIR / "julia_metal"

#: Full path to a Julia executable; wins over PATH discovery.
JULIA_ENV_VAR = "HORNLAB_BEAT_JULIA"
#: "1" reports the CPU backend as available. Internal/testing only: WG's
#: user-facing BEAT engine is GPU-only, and this is the switch its CI and
#: this package's own validation use to exercise the full plumbing without
#: accelerator hardware.
FORCE_CPU_ENV_VAR = "HORNLAB_BEAT_FORCE_CPU"

_PROBE_TIMEOUT_S = 300.0


def package_version() -> str | None:
    try:
        return importlib.metadata.version("hornlab-beat-bem")
    except importlib.metadata.PackageNotFoundError:
        return None


#: Files whose contents decide what a worker computes. Globs are relative to
#: the package directory and are resolved in sorted order.
_FINGERPRINT_GLOBS = (
    "*.py",
    "julia/*.jl",
    "julia/src/*.jl",
    "julia_engine/*/src/*.jl",
)


@lru_cache(maxsize=1)
def package_fingerprint() -> str:
    """A content hash of the wrapper and the vendored engine.

    ``package_version()`` is the obvious staleness signal and it is not enough
    here: consumers pin this repository **by commit SHA**, so the declared
    version sits still for a whole release cycle while the solver underneath
    it changes. A worker keyed on the version alone would therefore be adopted
    across a re-vendor, a driver fix or a wrapper change, and would answer
    every request with the previous code -- correctly formed numbers from a
    solver that no longer exists in the checkout. Since 0.3.2 a worker outlives
    the process that started it, so that adoption is now possible in the field
    and not only in an editable development tree.

    Content, not mtimes: a checkout, a copy and a restore all move mtimes
    without changing behaviour, and a patch applied in place can leave one
    alone. It costs 7.5 ms on an M1 Max -- 1.2 MB over 60 files -- once per
    process, against the 15 s it is protecting.
    """

    digest = hashlib.sha256()
    for pattern in _FINGERPRINT_GLOBS:
        for path in sorted(PACKAGE_DIR.glob(pattern)):
            digest.update(path.name.encode("utf-8"))
            try:
                digest.update(path.read_bytes())
            except OSError:
                digest.update(b"<unreadable>")
    return digest.hexdigest()[:16]


def default_project(beat_backend: str) -> Path:
    if beat_backend == BEAT_CPU:
        return CPU_PROJECT
    if beat_backend == BEAT_CUDA:
        return CUDA_PROJECT
    if beat_backend == BEAT_ROCM:
        return ROCM_PROJECT
    if beat_backend == BEAT_METAL:
        return METAL_PROJECT
    raise ValueError(f"Unknown BEAT backend: {beat_backend}")


def discover_julia(explicit: str | None = None) -> str | None:
    """Resolve Julia: explicit > env var > provisioned runtime > PATH.

    The provisioned runtime (a known-good portable Julia fetched by
    ``hornlab_beat_bem.provision`` on GPU hosts) outranks whatever happens to
    be on PATH, but an explicit path or the env var always wins.
    """

    for candidate in (explicit, os.environ.get(JULIA_ENV_VAR)):
        if candidate and candidate.strip():
            path = Path(candidate.strip())
            if path.exists():
                return str(path)
            # A configured-but-wrong path is a misconfiguration to surface,
            # not something to silently fall past.
            return None
    from .provision import provisioned_julia

    provisioned = provisioned_julia()
    if provisioned is not None:
        return provisioned
    return shutil.which("julia")


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}" if str(exc) else type(exc).__name__


def _nvidia_gpu_present() -> bool:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False
    try:
        return subprocess.run(
            [executable, "-L"], capture_output=True, timeout=15.0, check=False
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _apple_gpu_present() -> bool:
    """Whether this host could have a Metal device at all.

    Metal.jl needs Apple Silicon: the package has no Intel-Mac path, and
    ``Metal.functional()`` is the authority anyway. This is only the cheap
    inventory check that decides whether paying for a Julia startup is worth
    it, exactly as ``nvidia-smi`` is for CUDA.
    """

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _rocm_present() -> bool:
    for env_name in ("BLAB_ROCM_PATH", "ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
        value = os.environ.get(env_name, "").strip()
        if value and Path(value).is_dir():
            return True
    return any(shutil.which(name) for name in ("rocminfo", "hipinfo", "hipInfo"))


@lru_cache(maxsize=4)
def _julia_gpu_functional(julia: str, beat_backend: str) -> tuple[bool, str]:
    """Ask Julia itself whether the accelerator package can actually run.

    This is the same question the first solve would ask, so a truthful READY
    here cannot be followed by a mid-solve accelerator failure on load. It is
    slow (a Julia startup plus CUDA/AMDGPU load), so it only runs after the
    cheap host inventory says the hardware family exists, and the answer is
    cached for the process lifetime.
    """

    module = {BEAT_CUDA: "CUDA", BEAT_ROCM: "AMDGPU", BEAT_METAL: "Metal"}[beat_backend]
    project = default_project(beat_backend)
    if not (project / "Project.toml").exists():
        return False, f"bundled Julia project {project} is missing Project.toml"
    command = [
        julia,
        f"--project={project}",
        "--startup-file=no",
        "-e",
        f"import {module}; exit({module}.functional() ? 0 : 1)",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Julia {module} probe could not run ({_describe(exc)})"
    if completed.returncode == 0:
        return True, f"Julia {module}.functional() confirmed a usable device"
    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    tail = detail[-1][:200] if detail else f"exit code {completed.returncode}"
    return False, (
        f"{module}.functional() is false ({tail}). Instantiate the "
        f"environment with: julia --project={project} -e \"using Pkg; Pkg.instantiate()\""
    )


def probe_gpu_functional_cache_clear() -> None:
    _julia_gpu_functional.cache_clear()


def beat_engine_status(*, julia_executable: str | None = None) -> dict[str, Any]:
    """Report whether a BEAT solve can run here, and on which backend.

    The user-facing answer is GPU-only: the CPU path exists as internal
    scaffolding and is reported available only under ``HORNLAB_BEAT_FORCE_CPU``.
    """

    version = package_version()
    julia = discover_julia(julia_executable)
    if julia is None:
        return {
            "available": False,
            "reason": (
                "No Julia executable was found. On a machine with an NVIDIA "
                "GPU, run: python -m hornlab_beat_bem.provision (downloads "
                "Julia and the CUDA stack). Otherwise install Julia >= 1.10 "
                f"and put it on PATH or set {JULIA_ENV_VAR}."
            ),
            "version": version,
            "backend": None,
            "julia_executable": None,
        }
    if not DEFAULT_SOLVER_SCRIPT.exists():
        return {
            "available": False,
            "reason": f"Bundled BEAT solver script is missing: {DEFAULT_SOLVER_SCRIPT}",
            "version": version,
            "backend": None,
            "julia_executable": julia,
        }

    if os.environ.get(FORCE_CPU_ENV_VAR, "").strip() == "1":
        return {
            "available": True,
            "reason": (
                f"CPU backend forced by {FORCE_CPU_ENV_VAR}=1 (internal "
                "scaffolding/regression path; not a user-facing WG backend)"
            ),
            "version": version,
            "backend": BEAT_CPU,
            "julia_executable": julia,
        }

    if _nvidia_gpu_present():
        from .provision import read_state

        provisioning = read_state() or {}
        if provisioning.get("status") == "in_progress":
            return {
                "available": False,
                "reason": (
                    "BEAT CUDA runtime provisioning is in progress "
                    f"(step: {provisioning.get('step', 'unknown')}). The engine "
                    "becomes available when it finishes."
                ),
                "version": version,
                "backend": None,
                "julia_executable": julia,
            }
        functional, detail = _julia_gpu_functional(julia, BEAT_CUDA)
        if functional:
            return {
                "available": True,
                "reason": f"NVIDIA GPU detected and {detail}",
                "version": version,
                "backend": BEAT_CUDA,
                "julia_executable": julia,
            }
        if provisioning.get("status") == "failed":
            detail = (
                f"provisioning failed earlier: {provisioning.get('error')}. "
                "Retry with: python -m hornlab_beat_bem.provision --force"
            )
        return {
            "available": False,
            "reason": f"An NVIDIA GPU is present but the CUDA path is not usable: {detail}",
            "version": version,
            "backend": None,
            "julia_executable": julia,
        }

    if _rocm_present():
        from .provision import read_state

        provisioning = read_state() or {}
        if provisioning.get("status") == "in_progress":
            return {
                "available": False,
                "reason": (
                    "BEAT ROCm runtime provisioning is in progress "
                    f"(step: {provisioning.get('step', 'unknown')}). The engine "
                    "becomes available when it finishes."
                ),
                "version": version,
                "backend": None,
                "julia_executable": julia,
            }
        functional, detail = _julia_gpu_functional(julia, BEAT_ROCM)
        if functional:
            return {
                "available": True,
                "reason": f"AMD ROCm runtime detected and {detail}",
                "version": version,
                "backend": BEAT_ROCM,
                "julia_executable": julia,
            }
        if provisioning.get("status") == "failed":
            detail = (
                f"provisioning failed earlier: {provisioning.get('error')}. "
                "Retry with: python -m hornlab_beat_bem.provision --backend rocm --force"
            )
        return {
            "available": False,
            "reason": f"An AMD ROCm runtime is present but the path is not usable: {detail}",
            "version": version,
            "backend": None,
            "julia_executable": julia,
        }

    if _apple_gpu_present():
        functional, detail = _julia_gpu_functional(julia, BEAT_METAL)
        if functional:
            return {
                "available": True,
                "reason": f"Apple Silicon GPU detected and {detail}",
                "version": version,
                "backend": BEAT_METAL,
                "julia_executable": julia,
            }
        return {
            "available": False,
            "reason": f"This is Apple Silicon but the Metal path is not usable: {detail}",
            "version": version,
            "backend": None,
            "julia_executable": julia,
        }

    return {
        "available": False,
        "reason": (
            "No supported GPU was detected (no NVIDIA CUDA device via "
            "nvidia-smi, no AMD ROCm runtime, and not Apple Silicon). The BEAT "
            "engine is GPU-only; the internal CPU path is enabled by "
            f"{FORCE_CPU_ENV_VAR}=1 for tests and validation."
        ),
        "version": version,
        "backend": None,
        "julia_executable": julia,
    }
