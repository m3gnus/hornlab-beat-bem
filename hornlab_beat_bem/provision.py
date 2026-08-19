"""GPU-gated runtime provisioning: fetch Julia and the CUDA stack on demand.

Nothing here downloads anything unless an NVIDIA GPU is actually present:
``provision_cuda`` (and the ``--if-nvidia-gpu`` CLI gate) checks the hardware
inventory first and exits as a no-op otherwise, so CPU-only machines never pay
the ~200 MB Julia + multi-GB CUDA.jl artifact cost.

Steps, each idempotent and recorded in ``<runtime_dir>/state.json``:

1. Resolve a Julia executable -- an existing install (env var/PATH/previous
   provisioning) wins; only when none exists is the official portable Julia
   downloaded, SHA-256 verified, and unpacked under the runtime directory.
2. ``Pkg.instantiate()`` the bundled ``julia_cuda`` project (downloads
   CUDA.jl and friends into the user's Julia depot).
3. Force CUDA artifact resolution and require ``CUDA.functional()``, so a
   recorded "ready" state means the first real solve will not stall on
   downloads or discover a broken driver.

Failures never raise past the CLI: they are written to the state file and
reported through ``beat_engine_status`` as honest unavailability reasons.
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

from .config import BEAT_CUDA
from .runtime import CUDA_PROJECT

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
}

#: Rough disk requirement for Julia + depot + CUDA artifacts.
_REQUIRED_FREE_BYTES = 6 * 1024**3

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


def _ensure_julia(runtime_dir: Path, status_cb: StatusCallback) -> str:
    """Resolve Julia, downloading the portable build only when none exists."""

    from .runtime import discover_julia

    existing = discover_julia()
    if existing is not None:
        status_cb(f"Using existing Julia: {existing}")
        return existing

    key = (platform.system(), platform.machine())
    download = _JULIA_DOWNLOADS.get(key)
    if download is None:
        raise RuntimeError(
            f"No portable Julia download is configured for {key[0]}/{key[1]}; "
            "install Julia manually and set HORNLAB_BEAT_JULIA."
        )
    free = shutil.disk_usage(runtime_dir.parent if runtime_dir.parent.exists() else Path.home()).free
    if free < _REQUIRED_FREE_BYTES:
        raise RuntimeError(
            f"Not enough free disk space for the GPU runtime: {free / 1e9:.1f} GB free, "
            f"~{_REQUIRED_FREE_BYTES / 1e9:.0f} GB needed."
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
    label: str,
    status_cb: StatusCallback,
) -> None:
    command = [julia, f"--project={CUDA_PROJECT}", "--startup-file=no", "-e", code]
    status_cb(label)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "BLAB_BEAT_ENGINE_GPU_BACKEND": BEAT_CUDA},
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


def provision_cuda(
    runtime_dir: Path | None = None,
    *,
    status_cb: StatusCallback = print,
    force: bool = False,
) -> dict[str, Any]:
    """Provision the CUDA runtime iff an NVIDIA GPU is present.

    Returns the resulting state dict; ``status`` is one of ``skipped`` (no
    GPU -- nothing was downloaded), ``ready``, or ``failed``.
    """

    from . import runtime

    directory = (runtime_dir or default_runtime_dir()).expanduser()

    if not runtime._nvidia_gpu_present():
        status_cb("No NVIDIA GPU detected; skipping BEAT CUDA runtime provisioning.")
        return {"status": "skipped", "reason": "no NVIDIA GPU detected"}

    previous = read_state(directory)
    if not force and previous and previous.get("status") == "ready":
        executable = previous.get("julia_executable")
        if isinstance(executable, str) and Path(executable).exists():
            status_cb("BEAT CUDA runtime is already provisioned.")
            return previous

    state: dict[str, Any] = {
        "status": "in_progress",
        "backend": BEAT_CUDA,
        "step": "resolve_julia",
        "julia_version": JULIA_VERSION,
    }
    _write_state(directory, state)
    try:
        julia = _ensure_julia(directory, status_cb)
        state.update(julia_executable=julia, step="instantiate")
        _write_state(directory, state)
        _run_julia_step(
            julia,
            'using Pkg; Pkg.instantiate()',
            label="Instantiating the Julia CUDA environment (first run downloads several GB)",
            status_cb=status_cb,
        )
        state["step"] = "cuda_probe"
        _write_state(directory, state)
        # versioninfo() forces CUDA artifact resolution, so a "ready" state
        # means the first solve starts computing instead of downloading.
        _run_julia_step(
            julia,
            'import CUDA; CUDA.versioninfo(); exit(CUDA.functional() ? 0 : 1)',
            label="Resolving CUDA artifacts and probing the device",
            status_cb=status_cb,
        )
        state.update(status="ready", step="done", error=None)
        _write_state(directory, state)
        runtime.probe_gpu_functional_cache_clear()
        status_cb("BEAT CUDA runtime is ready.")
        return state
    except Exception as exc:  # noqa: BLE001 - recorded, reported, never silent
        state.update(status="failed", error=str(exc))
        _write_state(directory, state)
        status_cb(f"BEAT CUDA runtime provisioning failed: {exc}")
        return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Provision the BEAT Engine CUDA runtime (GPU-gated; a "
        "no-op without an NVIDIA GPU).",
    )
    parser.add_argument(
        "--if-nvidia-gpu",
        action="store_true",
        help="exit 0 quietly when no NVIDIA GPU is present (for setup hooks)",
    )
    parser.add_argument("--force", action="store_true", help="re-provision even when ready")
    parser.add_argument("--dir", type=Path, default=None, help="runtime directory override")
    args = parser.parse_args(argv)

    from . import runtime

    if args.if_nvidia_gpu and not runtime._nvidia_gpu_present():
        return 0
    state = provision_cuda(args.dir, force=args.force)
    if state["status"] == "ready":
        return 0
    if state["status"] == "skipped":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
