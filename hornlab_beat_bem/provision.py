"""Runtime provisioning: fetch Julia and one backend's stack on demand.

Two entry points, and the difference between them is who decides:

``provision_gpu`` is **hardware-gated and stays that way.** It downloads
nothing unless matching GPU hardware is actually present -- it (and the
``--if-gpu`` CLI gate) checks the hardware inventory first and exits as a
no-op otherwise, so a CPU-only machine that merely *ran* the setup hook never
pays the ~275 MB Julia + multi-GB CUDA.jl/AMDGPU.jl artifact cost.

``provision_cpu`` is **explicit opt-in and has no gate at all.** An installer
on a GPU-less Windows or Linux host has to be able to say "provision BEAT for
this CPU" and get a working runtime without a GPU and without the operator
hand-setting ``HORNLAB_BEAT_JULIA``; nothing infers it, so the promise above
is unchanged for everyone who does not ask. ``--backend cpu`` is the CLI form,
``--backend auto`` never selects it, and combining it with a GPU gate flag is
refused rather than resolved (see ``main``).

Steps, each idempotent and recorded in ``<runtime_dir>/state.json``:

1. Resolve a Julia executable -- an existing install (env var/PATH/previous
   provisioning, including this runtime directory's own record) wins; only
   when none exists is the official portable Julia downloaded, SHA-256
   verified, and unpacked under the runtime directory.
2. ``Pkg.instantiate()`` the bundled backend project -- ``julia_cuda``,
   ``julia_rocm``, ``julia_metal``, or ``julia`` for the CPU. The CPU project
   depends on no accelerator package, so instantiating it pulls no GPU
   artifacts.
3. Probe. For a GPU that is artifact resolution plus ``functional()``; for the
   CPU it is a real 1 kHz solve through the precompiled engine bundle, which
   is the only thing that can tell a live bundle from the silent
   compile-from-source fallback (see ``AGENTS.md``, *The failure mode to watch
   for*). Either way a recorded "ready" means the first real solve computes
   instead of downloading or compiling.

The state file records the backend, project and content fingerprint it was
provisioned for, and both entry points refuse to reuse a "ready" record that
does not match what is being asked for: a CPU-ready runtime is never reported
as satisfying a GPU request, or the other way round, and an in-place package
update is instantiated and probed again.

Failures never raise past the CLI: they are written to the state file,
reported through ``beat_engine_status`` as honest unavailability reasons, and
turned into a nonzero exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import BEAT_CPU, BEAT_CUDA, BEAT_METAL, BEAT_ROCM

JULIA_VERSION = "1.12.6"
_JULIA_BASE = "https://julialang-s3.julialang.org/bin"
_CHECKSUMS_URL = f"{_JULIA_BASE}/checksums/julia-{JULIA_VERSION}.sha256"

#: filename, URL, and (when known ahead of time) pinned SHA-256 per platform.
#: The Windows hash is pinned from the official artifact; other platforms are
#: verified against the official checksums file fetched from the same origin.
_JULIA_DOWNLOADS: dict[tuple[str, str], dict[str, str | None]] = {
    ("Windows", "AMD64"): {
        "filename": f"julia-{JULIA_VERSION}-win64.zip",
        "url": f"{_JULIA_BASE}/winnt/x64/1.12/julia-{JULIA_VERSION}-win64.zip",
        "sha256": "a63d991976e6893f508c512e3dc7bca1836c1a1f6ad1f3e4aedec159b6733e89",
    },
    ("Linux", "x86_64"): {
        "filename": f"julia-{JULIA_VERSION}-linux-x86_64.tar.gz",
        "url": f"{_JULIA_BASE}/linux/x64/1.12/julia-{JULIA_VERSION}-linux-x86_64.tar.gz",
        "sha256": None,
    },
    ("Darwin", "arm64"): {
        "filename": f"julia-{JULIA_VERSION}-macaarch64.tar.gz",
        "url": f"{_JULIA_BASE}/mac/aarch64/1.12/julia-{JULIA_VERSION}-macaarch64.tar.gz",
        "sha256": None,
    },
}

#: Rough disk requirement for Julia + depot + CUDA artifacts.
_REQUIRED_FREE_BYTES = 6 * 1024**3

#: The CPU project pulls no accelerator artifacts, so demanding the GPU figure
#: would refuse to provision hosts that have room for everything CPU needs:
#: the portable Julia unpacks to ~1.5 GB and the CPU depot is small.
_CPU_REQUIRED_FREE_BYTES = 2 * 1024**3

RUNTIME_DIR_ENV_VAR = "HORNLAB_BEAT_RUNTIME_DIR"
STATE_FILENAME = "state.json"

StatusCallback = Callable[[str], None]


def default_runtime_dir() -> Path:
    """Per-user runtime root; overridable with HORNLAB_BEAT_RUNTIME_DIR."""

    override = os.environ.get(RUNTIME_DIR_ENV_VAR, "").strip()
    if override:
        return Path(override)
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "hornlab-beat" / "runtime"
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "hornlab-beat" / "runtime"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "hornlab-beat" / "runtime"


def read_state(runtime_dir: Path | None = None) -> dict[str, Any] | None:
    path = (runtime_dir or default_runtime_dir()) / STATE_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_state(runtime_dir: Path, state: dict[str, Any]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = runtime_dir / STATE_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def provisioned_julia(runtime_dir: Path | None = None) -> str | None:
    """Return the provisioned Julia executable when it is recorded and real."""

    state = read_state(runtime_dir)
    if not state or state.get("status") != "ready":
        return None
    executable = state.get("julia_executable")
    if isinstance(executable, str) and executable and Path(executable).exists():
        return executable
    return None


def _download(url: str, destination: Path, *, expected_sha256: str | None, status_cb: StatusCallback) -> None:
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        last_report = 0.0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
            read += len(chunk)
            now = time.monotonic()
            if now - last_report >= 5.0:
                last_report = now
                if total:
                    status_cb(f"Downloading {destination.name}: {read / 1e6:.0f} / {total / 1e6:.0f} MB")
                else:
                    status_cb(f"Downloading {destination.name}: {read / 1e6:.0f} MB")
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    os.replace(tmp, destination)


def _official_checksum(filename: str, status_cb: StatusCallback) -> str:
    status_cb(f"Fetching official checksum for {filename}")
    with urllib.request.urlopen(_CHECKSUMS_URL) as response:
        text = response.read().decode("utf-8", errors="replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].strip() == filename:
            return parts[0].strip().lower()
    raise RuntimeError(f"{_CHECKSUMS_URL} does not list {filename}")


def _extract_julia(archive: Path, runtime_dir: Path, status_cb: StatusCallback) -> Path:
    """Unpack the portable Julia archive and return its executable."""

    status_cb(f"Unpacking {archive.name}")
    staging = Path(tempfile.mkdtemp(prefix="julia-unpack-", dir=str(runtime_dir)))
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(staging)
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(staging, filter="data")
        entries = [entry for entry in staging.iterdir() if entry.is_dir()]
        if len(entries) != 1:
            raise RuntimeError(f"unexpected archive layout in {archive.name}: {entries}")
        target = runtime_dir / entries[0].name
        if target.exists():
            shutil.rmtree(target)
        os.replace(entries[0], target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    executable = target / "bin" / ("julia.exe" if platform.system() == "Windows" else "julia")
    if not executable.exists():
        raise RuntimeError(f"unpacked Julia has no executable at {executable}")
    return executable


def _ensure_julia(
    runtime_dir: Path,
    status_cb: StatusCallback,
    *,
    required_bytes: int = _REQUIRED_FREE_BYTES,
    purpose: str = "GPU runtime",
    previous_executable: str | None = None,
) -> str:
    """Resolve Julia, downloading the portable build only when none exists.

    ``previous_executable`` is what an earlier run recorded in *this* runtime
    directory. It is a fallback rather than a first choice -- ``discover_julia``
    still wins, so an operator who repoints ``HORNLAB_BEAT_JULIA`` gets the
    Julia they named -- but it has to exist, because the in-progress state
    written before this call has already overwritten that record, and because
    ``discover_julia`` only ever reads the *default* runtime directory.
    Without it, ``--force`` and every re-run into a ``--dir`` of its own
    re-downloaded a Julia the directory already held.
    """

    from .runtime import discover_julia

    existing = discover_julia()
    if existing is not None:
        status_cb(f"Using existing Julia: {existing}")
        return existing

    if previous_executable and Path(previous_executable).exists():
        status_cb(f"Reusing the Julia this runtime directory already holds: {previous_executable}")
        return previous_executable

    key = (platform.system(), platform.machine())
    download = _JULIA_DOWNLOADS.get(key)
    if download is None:
        raise RuntimeError(
            f"No portable Julia download is configured for {key[0]}/{key[1]}; "
            "install Julia manually and set HORNLAB_BEAT_JULIA."
        )
    free = shutil.disk_usage(runtime_dir.parent if runtime_dir.parent.exists() else Path.home()).free
    if free < required_bytes:
        raise RuntimeError(
            f"Not enough free disk space for the {purpose}: {free / 1e9:.1f} GB free, "
            f"~{required_bytes / 1e9:.0f} GB needed."
        )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    expected = download["sha256"] or _official_checksum(str(download["filename"]), status_cb)
    archive = runtime_dir / str(download["filename"])
    status_cb(f"Downloading portable Julia {JULIA_VERSION} (~{275_000_000 / 1e6:.0f} MB)")
    _download(str(download["url"]), archive, expected_sha256=expected, status_cb=status_cb)
    executable = _extract_julia(archive, runtime_dir, status_cb)
    archive.unlink(missing_ok=True)
    return str(executable)


def _run_julia_step(
    julia: str,
    code: str,
    *,
    project: Path,
    env_backend: str,
    label: str,
    status_cb: StatusCallback,
    extra_env: dict[str, str] | None = None,
) -> None:
    command = [julia, f"--project={project}", "--startup-file=no", "-e", code]
    status_cb(label)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "BLAB_BEAT_ENGINE_GPU_BACKEND": env_backend,
            **(extra_env or {}),
        },
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            tail.append(text)
            del tail[:-15]
            status_cb(text)
    if process.wait() != 0:
        detail = "\n".join(tail[-10:])
        raise RuntimeError(f"{label} failed (exit {process.returncode}).\n{detail}")


#: Per-backend provisioning facts: hardware gate, Julia project, and module.
_GPU_BACKENDS: dict[str, dict[str, Any]] = {
    BEAT_CUDA: {
        "label": "CUDA",
        "module": "CUDA",
        "hardware": "an NVIDIA GPU",
    },
    BEAT_ROCM: {
        "label": "ROCm",
        "module": "AMDGPU",
        "hardware": "an AMD ROCm runtime",
    },
    BEAT_METAL: {
        "label": "Metal",
        "module": "Metal",
        "hardware": "an Apple Silicon GPU",
        # Metal.jl ships no vendor toolkit: the driver is the operating
        # system's, so instantiating is a small package download rather than
        # the multi-gigabyte artifact pull CUDA and ROCm need.
        "instantiate_note": "",
    },
}


def _recorded_julia(previous: dict[str, Any] | None) -> str | None:
    """The Julia executable a previous run of *this* runtime directory recorded.

    Any status will do -- an interrupted or failed run still names a Julia it
    successfully unpacked, and re-downloading it would be pure waste. What it
    must not do is make the *runtime* look ready; that is ``provisioned_julia``,
    which requires ``status == "ready"``.
    """

    if not previous:
        return None
    executable = previous.get("julia_executable")
    if isinstance(executable, str) and executable and Path(executable).exists():
        return executable
    return None


def _ready_for(
    previous: dict[str, Any] | None,
    backend: str,
    project: Path,
    fingerprint: str,
) -> bool:
    """Whether a recorded state is a ready runtime for exactly this request.

    Identity is the backend *and* the project path. The backend check is what
    keeps a CPU-ready runtime from being reported as satisfying a CUDA request
    and the reverse. The project check catches a package that moved -- a
    reinstall into a different prefix leaves the recorded Julia in place while
    the bundled project it instantiated is gone. A legacy record that predates
    the key is re-provisioned: dependency and source identity cannot be
    inferred safely from an old record.

    The fingerprint is content-based and cached for the Python process
    lifetime. An installed-package update starts a new process; tests that
    simulate one in place must clear ``runtime.package_fingerprint`` first.
    """

    if not previous or previous.get("status") != "ready":
        return False
    if previous.get("backend") != backend:
        return False
    recorded_project = previous.get("project")
    if recorded_project != str(project):
        return False
    if previous.get("package_fingerprint") != fingerprint:
        return False
    return _recorded_julia(previous) is not None


def detect_gpu_backend() -> str | None:
    """Which GPU backend this host's hardware inventory suggests, if any."""

    from . import runtime

    if runtime._nvidia_gpu_present():
        return BEAT_CUDA
    if runtime._rocm_present():
        return BEAT_ROCM
    if runtime._apple_gpu_present():
        return BEAT_METAL
    return None


def _gpu_hardware_present(backend: str) -> bool:
    from . import runtime

    if backend == BEAT_CUDA:
        return runtime._nvidia_gpu_present()
    if backend == BEAT_ROCM:
        return runtime._rocm_present()
    if backend == BEAT_METAL:
        return runtime._apple_gpu_present()
    return False


def provision_gpu(
    runtime_dir: Path | None = None,
    *,
    backend: str = BEAT_CUDA,
    status_cb: StatusCallback = print,
    force: bool = False,
) -> dict[str, Any]:
    """Provision a GPU runtime iff the matching hardware is present.

    Returns the resulting state dict; ``status`` is one of ``skipped`` (no
    matching GPU -- nothing was downloaded), ``ready``, or ``failed``.
    """

    from . import runtime
    from .runtime import default_project, package_fingerprint

    if backend not in _GPU_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_GPU_BACKENDS)}")
    facts = _GPU_BACKENDS[backend]
    directory = (runtime_dir or default_runtime_dir()).expanduser()

    if not _gpu_hardware_present(backend):
        status_cb(
            f"No {facts['hardware']} detected; skipping BEAT {facts['label']} "
            "runtime provisioning."
        )
        return {"status": "skipped", "reason": f"no {facts['hardware']} detected"}

    project = default_project(backend)
    fingerprint = package_fingerprint(project)
    previous = read_state(directory)
    if not force and _ready_for(previous, backend, project, fingerprint):
        assert previous is not None
        status_cb(f"BEAT {facts['label']} runtime is already provisioned.")
        return previous

    module = facts["module"]
    state: dict[str, Any] = {
        "status": "in_progress",
        "backend": backend,
        "project": str(project),
        "package_fingerprint": fingerprint,
        "step": "resolve_julia",
        "julia_version": JULIA_VERSION,
    }
    _write_state(directory, state)
    try:
        julia = _ensure_julia(
            directory, status_cb, previous_executable=_recorded_julia(previous)
        )
        state.update(julia_executable=julia, step="instantiate")
        _write_state(directory, state)
        _run_julia_step(
            julia,
            'using Pkg; Pkg.instantiate()',
            project=project,
            env_backend=backend,
            label=(
                f"Instantiating the Julia {facts['label']} environment"
                + facts.get("instantiate_note", " (first run downloads several GB)")
            ),
            status_cb=status_cb,
        )
        state["step"] = "gpu_probe"
        _write_state(directory, state)
        # versioninfo() forces artifact resolution, so a "ready" state means
        # the first solve starts computing instead of downloading.
        _run_julia_step(
            julia,
            f'import {module}; {module}.versioninfo(); exit({module}.functional() ? 0 : 1)',
            project=project,
            env_backend=backend,
            label=f"Resolving {facts['label']} artifacts and probing the device",
            status_cb=status_cb,
        )
        state.update(status="ready", step="done", error=None)
        _write_state(directory, state)
        runtime.probe_gpu_functional_cache_clear()
        status_cb(f"BEAT {facts['label']} runtime is ready.")
        return state
    except Exception as exc:  # noqa: BLE001 - recorded, reported, never silent
        state.update(status="failed", error=str(exc))
        _write_state(directory, state)
        status_cb(f"BEAT {facts['label']} runtime provisioning failed: {exc}")
        return state


#: The CPU probe: instantiating is not evidence, so solve something.
#:
#: This asks the *precompiled bundle* to run one 1 kHz solve of its own
#: workload mesh and checks three separate things a mere ``Pkg.instantiate()``
#: cannot: that the bundle imports at all, that it resolved this package's
#: engine directory rather than an adjacent checkout, and that the solve
#: reaches a finite non-zero pressure row. The engine-directory check is not
#: paranoia -- ``ENGINE_DIR`` is a relative search, and a bundle that finds a
#: foreign one precompiles code this package will not run.
_CPU_PROBE_ENGINE_DIR_ENV_VAR = "HORNLAB_BEAT_PROBE_ENGINE_DIR"
#:
#: The body is one function rather than top-level statements on purpose: at
#: top level a ``for`` loop that assigns to an outer name lands in Julia's
#: soft-scope rule, and the probe's verdict would be computed into a local
#: nobody reads.
_CPU_PROBE_CODE = """
using BeatEngineCpuBundle
using JSON
const B = BeatEngineCpuBundle

function beat_cpu_probe()
    expected = realpath(ENV["HORNLAB_BEAT_PROBE_ENGINE_DIR"])
    found = realpath(String(B.ENGINE_DIR))
    if !samefile(found, expected)
        println(stderr, "engine bundle resolved ", found, ", not this package's ", expected)
        return 2
    end
    work = mktempdir()
    mesh = joinpath(work, "probe.msh")
    write(mesh, B.WORKLOAD_MESH)
    request = Dict{String,Any}(
        "schema_version" => 2,
        "beat_engine_backend" => "cpu",
        "frequencies_hz" => [1000.0],
        "config" => Dict{String,Any}(
            "mesh_file" => mesh, "scale_factor" => 1.0, "distance" => 1.0,
            "axial_offset" => 0.0, "step_size" => 90.0, "min_angle" => 0.0,
            "max_angle" => 90.0, "freq_min" => 1000.0, "freq_max" => 1000.0,
            "freq_count" => 1, "tag_throat" => 2, "rho" => 1.2041,
            "sound_speed" => 343.0, "symmetry" => "off", "source_motion" => "normal"))
    events = joinpath(work, "events.jsonl")
    open(events, "w") do io
        redirect_stdout(io) do
            B.solve_request(request)
        end
    end
    results = 0
    usable = false
    completed = false
    for line in eachline(events)
        isempty(strip(line)) && continue
        event = JSON.parse(line)
        event_type = get(event, "type", "")
        if event_type == "result"
            results += 1
            pressure = event["result"]["horizontal_pressure"]
            real_rows = pressure["real"]
            imag_rows = pressure["imag"]
            real_values = (!isempty(real_rows) && real_rows[1] isa AbstractVector) ? real_rows[1] : real_rows
            imag_values = (!isempty(imag_rows) && imag_rows[1] isa AbstractVector) ? imag_rows[1] : imag_rows
            usable = (
                !isempty(real_values) && length(real_values) == length(imag_values) &&
                all(isfinite, real_values) && all(isfinite, imag_values) &&
                any(i -> !iszero(real_values[i]) || !iszero(imag_values[i]), eachindex(real_values)))
        elseif event_type == "completed"
            completed = get(event, "solved_count", 0) == 1
        end
    end
    if results != 1 || !usable || !completed
        println(stderr, "CPU solve probe returned ", results,
            " result events, usable=", usable, ", completed=", completed)
        return 3
    end
    println("CPU engine bundle solved a 1 kHz probe from ", found)
    return 0
end

exit(beat_cpu_probe())
"""


def provision_cpu(
    runtime_dir: Path | None = None,
    *,
    status_cb: StatusCallback = print,
    force: bool = False,
) -> dict[str, Any]:
    """Provision the CPU runtime. Explicit opt-in, never inferred.

    Unlike ``provision_gpu`` there is no hardware gate: the caller asked for
    the CPU, and a CPU is what every host has. This is the path an installer
    on a GPU-less Windows or Linux machine takes so that the package has a
    Julia to discover without the operator setting ``HORNLAB_BEAT_JULIA`` by
    hand -- ``discover_julia`` reads the recorded runtime, so a *ready* state
    here is what makes that tier live on a machine that has no GPU.

    Returns the resulting state dict; ``status`` is ``ready`` or ``failed``.
    There is no ``skipped``: nothing here can decline on the caller's behalf.
    """

    from .runtime import PACKAGE_DIR, default_project, package_fingerprint

    directory = (runtime_dir or default_runtime_dir()).expanduser()
    project = default_project(BEAT_CPU)
    fingerprint = package_fingerprint(project)
    previous = read_state(directory)
    if not force and _ready_for(previous, BEAT_CPU, project, fingerprint):
        assert previous is not None
        status_cb("BEAT CPU runtime is already provisioned.")
        return previous

    state: dict[str, Any] = {
        "status": "in_progress",
        "backend": BEAT_CPU,
        "project": str(project),
        "package_fingerprint": fingerprint,
        "step": "resolve_julia",
        "julia_version": JULIA_VERSION,
    }
    _write_state(directory, state)
    try:
        julia = _ensure_julia(
            directory,
            status_cb,
            required_bytes=_CPU_REQUIRED_FREE_BYTES,
            purpose="CPU runtime",
            previous_executable=_recorded_julia(previous),
        )
        state.update(julia_executable=julia, step="instantiate")
        _write_state(directory, state)
        _run_julia_step(
            julia,
            "using Pkg; Pkg.instantiate()",
            project=project,
            env_backend=BEAT_CPU,
            label="Instantiating the Julia CPU environment (no GPU artifacts)",
            status_cb=status_cb,
        )
        state["step"] = "cpu_probe"
        _write_state(directory, state)
        _run_julia_step(
            julia,
            _CPU_PROBE_CODE,
            project=project,
            env_backend=BEAT_CPU,
            label="Precompiling the CPU engine bundle and solving a probe frequency",
            status_cb=status_cb,
            extra_env={_CPU_PROBE_ENGINE_DIR_ENV_VAR: str(PACKAGE_DIR / "julia")},
        )
        state.update(status="ready", step="done", error=None)
        _write_state(directory, state)
        status_cb("BEAT CPU runtime is ready.")
        return state
    except Exception as exc:  # noqa: BLE001 - recorded, reported, never silent
        state.update(status="failed", error=str(exc))
        _write_state(directory, state)
        status_cb(f"BEAT CPU runtime provisioning failed: {exc}")
        return state


def provision_cuda(
    runtime_dir: Path | None = None,
    *,
    status_cb: StatusCallback = print,
    force: bool = False,
) -> dict[str, Any]:
    """Back-compat alias for ``provision_gpu(backend='cuda')``."""

    return provision_gpu(runtime_dir, backend=BEAT_CUDA, status_cb=status_cb, force=force)


def _note_undiscoverable_dir(directory: Path | None) -> None:
    """Say so when ``--dir`` writes a runtime nothing will ever discover.

    ``discover_julia`` resolves the provisioned tier through
    ``default_runtime_dir()``, which honours ``HORNLAB_BEAT_RUNTIME_DIR`` and
    nothing else. Provisioning into a directory that is not that one succeeds
    and then looks, from every later process, exactly like never having
    provisioned at all.
    """

    if directory is None:
        return
    resolved = directory.expanduser()
    if resolved == default_runtime_dir():
        return
    print(
        f"Note: BEAT discovers the provisioned runtime through "
        f"{RUNTIME_DIR_ENV_VAR} (currently {default_runtime_dir()}). Set "
        f"{RUNTIME_DIR_ENV_VAR}={resolved} in the environment that runs BEAT, "
        "or what is provisioned here will not be found."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Provision a BEAT Engine runtime. The default is the GPU "
        "stack this host's hardware suggests, and it is a no-op without a "
        "supported GPU; --backend cpu explicitly provisions the CPU runtime "
        "instead, on any host.",
    )
    parser.add_argument(
        "--if-gpu",
        action="store_true",
        help="exit 0 quietly when no supported GPU is present (for setup hooks)",
    )
    parser.add_argument(
        "--if-nvidia-gpu",
        action="store_true",
        help="exit 0 quietly when no NVIDIA GPU is present (legacy setup-hook gate)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", BEAT_CPU, BEAT_CUDA, BEAT_ROCM, BEAT_METAL],
        default="auto",
        help="which stack to provision; auto follows the GPU hardware "
        f"inventory and never selects {BEAT_CPU!r}, which is opt-in only",
    )
    parser.add_argument("--force", action="store_true", help="re-provision even when ready")
    parser.add_argument("--dir", type=Path, default=None, help="runtime directory override")
    args = parser.parse_args(argv)

    from . import runtime

    if args.backend == BEAT_CPU and (args.if_gpu or args.if_nvidia_gpu):
        # "Provision the CPU" and "do nothing unless there is a GPU" cannot
        # both be honoured, and either resolution silently does something the
        # caller did not ask for. Refuse instead of picking one: a setup hook
        # gets a legible error at authoring time rather than a runtime that is
        # sometimes provisioned.
        parser.error(
            "--backend cpu cannot be combined with --if-gpu/--if-nvidia-gpu: "
            "those gate on GPU hardware, and the CPU backend is an explicit "
            "opt-in that ignores it. Run them as two separate commands."
        )
    if args.if_nvidia_gpu and not runtime._nvidia_gpu_present():
        return 0
    if args.backend == BEAT_CPU:
        _note_undiscoverable_dir(args.dir)
        state = provision_cpu(args.dir, force=args.force)
        return 0 if state["status"] == "ready" else 1
    backend = args.backend
    if backend == "auto":
        backend = detect_gpu_backend()
        if backend is None:
            if args.if_gpu:
                return 0
            print(
                "No supported GPU (NVIDIA CUDA, AMD ROCm, or Apple Silicon "
                "Metal) was detected; nothing to provision."
            )
            return 0
    elif args.if_gpu and not _gpu_hardware_present(backend):
        return 0
    _note_undiscoverable_dir(args.dir)
    state = provision_gpu(args.dir, backend=backend, force=args.force)
    if state["status"] in {"ready", "skipped"}:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
