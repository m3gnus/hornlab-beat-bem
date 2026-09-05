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

from .config import BEAT_BACKENDS, BEAT_CPU, BEAT_CUDA, BEAT_METAL, BEAT_ROCM

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SOLVER_SCRIPT = PACKAGE_DIR / "julia" / "solver.jl"
CPU_PROJECT = PACKAGE_DIR / "julia"
CUDA_PROJECT = PACKAGE_DIR / "julia_cuda"
ROCM_PROJECT = PACKAGE_DIR / "julia_rocm"
METAL_PROJECT = PACKAGE_DIR / "julia_metal"

#: Full path to a Julia executable; wins over PATH discovery.
JULIA_ENV_VAR = "HORNLAB_BEAT_JULIA"
#: "1" reports the CPU backend as available **without consulting the
#: provisioning record**. It is the switch CI and this package's own validation
#: use to exercise the full plumbing on a host that has provisioned nothing and
#: has no accelerator.
#:
#: It is no longer the only way the CPU backend is reported available: a host
#: that has run ``provision --backend cpu`` has instantiated the project and
#: solved a 1 kHz probe, which is evidence this variable deliberately skips.
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
    "julia*/Project.toml",
    "julia*/Manifest.toml",
    "julia/src/*.jl",
    "julia_engine/*/Project.toml",
    "julia_engine/*/Manifest.toml",
    "julia_engine/*/src/*.jl",
)


def _hash_file(digest: Any, path: Path, identity: str) -> None:
    """Add one named file to a runtime identity without trusting its mtime."""

    name = identity.encode("utf-8")
    content = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                content.update(chunk)
    except OSError:
        content.update(b"<unreadable>")
    digest.update(len(name).to_bytes(4, "big"))
    digest.update(name)
    digest.update(content.digest())


@lru_cache(maxsize=16)
def package_fingerprint(
    julia_project: str | Path | None = None,
    julia_sysimage: str | Path | None = None,
) -> str:
    """A content hash of the wrapper and the selected Julia runtime inputs.

    ``package_version()`` is the obvious staleness signal and it is not enough
    here: consumers pin this repository **by commit SHA**, so the declared
    version sits still for a whole release cycle while the solver underneath
    it changes. A worker keyed on the version alone would therefore be adopted
    across a re-vendor, a driver fix or a wrapper change, and would answer
    every request with the previous code -- correctly formed numbers from a
    solver that no longer exists in the checkout. Since 0.3.2 a worker outlives
    the process that started it, so that adoption is now possible in the field
    and not only in an editable development tree.

    The backend environments and bundle projects are part of the package
    identity too. A Manifest can change the code Julia loads while every
    Python and Julia source file stays byte-identical. For the same reason an
    explicitly selected project and sysimage are hashed as runtime inputs;
    their resolved paths remain separate fields in the worker key.

    Content, not mtimes: a checkout, a copy and a restore all move mtimes
    without changing behaviour, and a patch applied in place can leave one
    alone. The result is cached per selected project/sysimage for the process
    lifetime. Large custom sysimages cost one streaming read, which is still
    small beside adopting an incompatible warm runtime.
    """

    digest = hashlib.sha256()
    for pattern in _FINGERPRINT_GLOBS:
        for path in sorted(PACKAGE_DIR.glob(pattern)):
            _hash_file(digest, path, path.relative_to(PACKAGE_DIR).as_posix())

    if julia_project is not None:
        project = Path(julia_project)
        project_root = (
            project.parent
            if project.name in {"Project.toml", "Manifest.toml"}
            else project
        )
        project_files = (
            project_root / "Project.toml",
            project_root / "Manifest.toml",
        )
        for path in project_files:
            _hash_file(digest, path, f"selected-project/{path.name}")

    if julia_sysimage is not None:
        _hash_file(digest, Path(julia_sysimage), "selected-sysimage")

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
    ``hornlab_beat_bem.provision``) outranks whatever happens to
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


def provision_command(backend: str) -> str:
    """The provisioning command for one backend, for an unavailable reason.

    Deliberately spelled ``python`` rather than resolved: the interpreter that
    has to run this is the one this package is installed into, which a consumer
    embedding the package knows and this module does not -- a packaged
    application keeps its Python off PATH entirely, so a resolved path here
    would be right in a checkout and wrong in the product. A consumer that can
    name its interpreter should say so in its own message.
    """

    return f"python -m hornlab_beat_bem.provision --backend {backend}"


#: How each GPU family is detected, in the words of an unavailable reason.
_GPU_INVENTORY: dict[str, tuple[str, str]] = {
    BEAT_CUDA: ("an NVIDIA GPU", "nvidia-smi -L reported no device"),
    BEAT_ROCM: (
        "an AMD ROCm runtime",
        "no rocminfo/hipinfo on PATH and no ROCM_PATH-style variable set",
    ),
    BEAT_METAL: ("an Apple Silicon GPU", "this host is not Apple Silicon"),
}

_GPU_HARDWARE_PRESENT: dict[str, Any] = {
    BEAT_CUDA: lambda: _nvidia_gpu_present(),
    BEAT_ROCM: lambda: _rocm_present(),
    BEAT_METAL: lambda: _apple_gpu_present(),
}


def _gpu_backend_status(backend: str, julia: str) -> tuple[bool, str, str]:
    """One accelerator's own verdict: available, reason, machine-readable state.

    Asked per backend rather than derived from a single "which backend would a
    solve use" answer. Those are different questions, and the difference is
    visible on any host with two accelerator families -- an NVIDIA card and an
    AMD card in one box -- where the single answer names one and says nothing
    true about the other.
    """

    from .provision import read_state

    hardware, how = _GPU_INVENTORY[backend]
    if not _GPU_HARDWARE_PRESENT[backend]():
        return False, f"No {hardware} was detected ({how}).", "no-hardware"

    provisioning = read_state(backend=backend) or {}
    if provisioning.get("status") == "in_progress" and backend != BEAT_METAL:
        return (
            False,
            (
                f"BEAT {backend} runtime provisioning is in progress "
                f"(step: {provisioning.get('step', 'unknown')}). The engine "
                "becomes available when it finishes."
            ),
            "provisioning",
        )
    functional, detail = _julia_gpu_functional(julia, backend)
    if functional:
        return True, f"{hardware.capitalize()} detected and {detail}", "ready"
    if provisioning.get("status") == "failed":
        detail = (
            f"provisioning failed earlier: {provisioning.get('error')}. "
            f"Retry with: {provision_command(backend)} --force"
        )
    return (
        False,
        f"{hardware.capitalize()} is present but the {backend} path is not usable: {detail}",
        "not-functional",
    )


def _cpu_backend_status(julia: str) -> tuple[bool, str, str]:
    """Whether a CPU solve can start here, from the provisioning record.

    A Julia executable and the bundled project on disk are not evidence: the
    project's checked-in Manifest names packages nothing may have downloaded,
    and the precompiled engine bundle may be absent, in which case the solve
    still answers correctly at three to five times the cold start with nothing
    logged (see ``AGENTS.md``). ``provision.provision_cpu`` instantiates the
    project and then *solves a 1 kHz probe* before recording ``ready``, so that
    record is the honest answer and this reads it rather than re-deriving one.

    ``HORNLAB_BEAT_FORCE_CPU=1`` still forces availability. It is what CI and
    this package's own validation use to exercise the plumbing on a host that
    has provisioned nothing.
    """

    from .provision import backend_ready, read_state

    if os.environ.get(FORCE_CPU_ENV_VAR, "").strip() == "1":
        return (
            True,
            (
                f"CPU backend forced by {FORCE_CPU_ENV_VAR}=1 (regression and "
                "CI path: no provisioning record was consulted)"
            ),
            "forced",
        )
    if backend_ready(BEAT_CPU):
        return (
            True,
            (
                "The BEAT CPU runtime is provisioned: its Julia project was "
                "instantiated and proved with a 1 kHz solve through the "
                f"precompiled engine bundle ({julia}). No accelerator needed."
            ),
            "ready",
        )
    recorded = read_state(backend=BEAT_CPU) or {}
    status = recorded.get("status")
    if status == "in_progress":
        return (
            False,
            (
                "BEAT CPU runtime provisioning is in progress (step: "
                f"{recorded.get('step', 'unknown')}). The engine becomes "
                "available when it finishes."
            ),
            "provisioning",
        )
    if status == "failed":
        return (
            False,
            (
                "BEAT CPU runtime provisioning failed here: "
                f"{recorded.get('error') or 'no reason was recorded'}. Retry "
                f"with: {provision_command(BEAT_CPU)} --force"
            ),
            "failed",
        )
    return (
        False,
        (
            "The BEAT CPU runtime has not been instantiated and probed here, "
            "and a Julia executable alone is not evidence that a solve would "
            "run -- an uninstantiated or offline depot fails at the first solve "
            f"instead. Run: {provision_command(BEAT_CPU)}. Readiness is "
            "recorded per backend, so this does not disturb a provisioned GPU "
            "runtime."
        ),
        "unprovisioned",
    )


def _no_julia_status(version: str | None) -> dict[str, Any]:
    return {
        "available": False,
        "reason": (
            "No Julia executable was found. Run: python -m "
            "hornlab_beat_bem.provision --backend cpu for the CPU runtime, "
            "or omit --backend cpu to provision a detected GPU. "
            f"Alternatively install Julia >= 1.10 and put it on PATH or set {JULIA_ENV_VAR}."
        ),
        "version": version,
        "backend": None,
        "julia_executable": None,
        "state": "no-julia",
    }


def _no_solver_script_status(version: str | None, julia: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": f"Bundled BEAT solver script is missing: {DEFAULT_SOLVER_SCRIPT}",
        "version": version,
        "backend": None,
        "julia_executable": julia,
        "state": "no-solver-script",
    }


def backend_status(backend: str, *, julia_executable: str | None = None) -> dict[str, Any]:
    """Whether *this* backend could run a solve here, and why not when it cannot.

    One backend, one answer, independent of every other backend. That
    independence is the point: a CUDA host can have a provisioned CPU runtime
    as well, and did not use to be able to say so -- readiness was one record
    with one ``backend`` field, so the two backends could not both be ready and
    an interface offering them as separate engines had to leave one permanently
    greyed out.

    ``state`` is the machine-readable half (``ready``, ``no-hardware``,
    ``not-functional``, ``provisioning``, ``failed``, ``unprovisioned``,
    ``forced``, ``no-julia``, ``no-solver-script``); ``reason`` is the sentence
    a user reads.
    """

    if backend not in BEAT_BACKENDS:
        raise ValueError(
            f"unknown BEAT backend {backend!r}; expected one of {', '.join(BEAT_BACKENDS)}"
        )
    version = package_version()
    julia = discover_julia(julia_executable)
    if julia is None:
        return {**_no_julia_status(version), "backend": backend}
    if not DEFAULT_SOLVER_SCRIPT.exists():
        return {**_no_solver_script_status(version, julia), "backend": backend}
    if backend == BEAT_CPU:
        available, reason, state = _cpu_backend_status(julia)
    else:
        available, reason, state = _gpu_backend_status(backend, julia)
    return {
        "available": available,
        "reason": reason,
        "state": state,
        "version": version,
        "backend": backend,
        "julia_executable": julia,
    }


def beat_backend_statuses(*, julia_executable: str | None = None) -> dict[str, Any]:
    """Every backend's own verdict, in the order a selector should show them.

    A GPU probe is a Julia startup, so this is not free -- but it is only paid
    for a family whose hardware the cheap inventory found, and
    ``_julia_gpu_functional`` caches its answer for the process lifetime.
    """

    return {backend: backend_status(backend, julia_executable=julia_executable) for backend in BEAT_BACKENDS}


def beat_engine_status(*, julia_executable: str | None = None) -> dict[str, Any]:
    """Report whether a BEAT solve can run here, and on which backend.

    The *one* backend an unqualified solve would use, which is a different
    question from :func:`backend_status`'s. It prefers a working accelerator,
    and falls back to the CPU runtime when one has been provisioned -- so a
    machine whose CUDA install is broken reports the CPU path it can actually
    solve on instead of reporting BEAT unavailable. ``HORNLAB_BEAT_FORCE_CPU=1``
    still short-circuits to the CPU backend.
    """

    version = package_version()
    julia = discover_julia(julia_executable)
    if julia is None:
        return _no_julia_status(version)
    if not DEFAULT_SOLVER_SCRIPT.exists():
        return _no_solver_script_status(version, julia)

    if os.environ.get(FORCE_CPU_ENV_VAR, "").strip() == "1":
        available, reason, state = _cpu_backend_status(julia)
        return {
            "available": available,
            "reason": reason,
            "state": state,
            "version": version,
            "backend": BEAT_CPU,
            "julia_executable": julia,
        }

    accelerator_reasons: list[str] = []
    for backend in (BEAT_CUDA, BEAT_ROCM, BEAT_METAL):
        available, reason, state = _gpu_backend_status(backend, julia)
        if available:
            return {
                "available": True,
                "reason": reason,
                "state": state,
                "version": version,
                "backend": backend,
                "julia_executable": julia,
            }
        if state != "no-hardware":
            # This host has the hardware and cannot use it. That is the reason
            # worth reporting; an absent family's is noise.
            accelerator_reasons.append(reason)

    cpu_available, cpu_reason, cpu_state = _cpu_backend_status(julia)
    if cpu_available:
        return {
            "available": True,
            "reason": cpu_reason,
            "state": cpu_state,
            "version": version,
            "backend": BEAT_CPU,
            "julia_executable": julia,
        }
    if accelerator_reasons:
        return {
            "available": False,
            "reason": " ".join(accelerator_reasons),
            "state": "not-functional",
            "version": version,
            "backend": None,
            "julia_executable": julia,
        }
    return {
        "available": False,
        "reason": (
            "No supported GPU was detected (no NVIDIA CUDA device via "
            "nvidia-smi, no AMD ROCm runtime, and not Apple Silicon), and the "
            f"portable CPU runtime is not provisioned here. {cpu_reason}"
        ),
        "state": cpu_state,
        "version": version,
        "backend": None,
        "julia_executable": julia,
    }
