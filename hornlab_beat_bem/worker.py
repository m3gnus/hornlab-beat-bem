"""Julia process management for the vendored BEAT Engine solver.

Ported from boundary-lab's ``blab/solvers/beat_engine_backend.py`` (GPL-3.0):
a persistent stdin/stdout JSON-lines worker per (executable, project, threads,
sysimage) key, a one-shot subprocess fallback, file-based cooperative
cancellation, and warm-up. The asset-staging layer is gone -- callers here
always hand the solver an on-disk mesh path directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .config import BEAT_CUDA, BEAT_METAL, BEAT_ROCM
from .runtime import DEFAULT_SOLVER_SCRIPT, default_project

StatusCallback = Callable[[str], None]

_WORKERS_LOCK = threading.Lock()
_WORKERS: dict[tuple[str, str, str, str, str], BeatWorkerProcess] = {}


def _resolve_julia_threads(julia_threads: str | int = "auto") -> str:
    if isinstance(julia_threads, int):
        return str(max(1, julia_threads))
    text = str(julia_threads or "auto").strip().lower()
    if text == "auto":
        return str(os.cpu_count() or 1)
    try:
        return str(max(1, int(text)))
    except ValueError:
        return str(os.cpu_count() or 1)


def _julia_process_env(julia_threads: str | int, julia_project: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = _resolve_julia_threads(julia_threads)
    if julia_project is not None:
        # The solver's accelerator hint falls back to the project directory
        # name; the env var makes the choice explicit even for custom paths.
        name = Path(julia_project).name.lower()
        if name == "julia_cuda":
            env["BLAB_BEAT_ENGINE_GPU_BACKEND"] = BEAT_CUDA
        elif name == "julia_rocm":
            env["BLAB_BEAT_ENGINE_GPU_BACKEND"] = BEAT_ROCM
        elif name == "julia_metal":
            env["BLAB_BEAT_ENGINE_GPU_BACKEND"] = BEAT_METAL
    return env


def _julia_command(
    julia_executable: str,
    solver_script: Path,
    *,
    julia_project: Path | None,
    julia_sysimage: Path | None,
    trailing: list[str],
) -> list[str]:
    command = [julia_executable]
    if julia_sysimage is not None:
        command.append(f"--sysimage={julia_sysimage}")
    if julia_project is not None:
        command.append(f"--project={julia_project}")
        command.append("--startup-file=no")
    command.append(str(solver_script))
    command.extend(trailing)
    return command


class BeatWorkerProcess:
    """One persistent Julia worker; one submission at a time."""

    def __init__(
        self,
        *,
        julia_executable: str,
        solver_script: Path,
        julia_threads: str | int,
        julia_project: Path | None,
        julia_sysimage: Path | None = None,
    ):
        self.julia_executable = julia_executable
        self.solver_script = solver_script
        self.julia_threads = julia_threads
        self.julia_project = julia_project
        self.julia_sysimage = julia_sysimage
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._status_callback: StatusCallback | None = None

    def submit(
        self,
        request_path: Path,
        *,
        status_callback: StatusCallback | None = None,
    ) -> Iterator[dict]:
        self._lock.acquire()
        self._status_callback = status_callback
        try:
            self._ensure_started()
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("Warm BEAT Engine solver did not provide stdin.")
            self._emit_status("Submitting solve request")
            process.stdin.write(
                json.dumps({"request": str(request_path), "operation": "solve"}, separators=(",", ":"))
                + "\n"
            )
            process.stdin.flush()
            return self._iter_events_for_submission()
        except Exception:
            self._status_callback = None
            self._lock.release()
            raise

    def ensure_started(self, *, status_callback: StatusCallback | None = None) -> None:
        with self._lock:
            previous = self._status_callback
            self._status_callback = status_callback
            try:
                self._ensure_started()
            finally:
                self._status_callback = previous

    def terminate(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if self._lock.locked():
            try:
                self._lock.release()
            except RuntimeError:
                pass

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._emit_status("BEAT Engine ready")
            return

        self._stderr_lines.clear()
        command = _julia_command(
            self.julia_executable,
            self.solver_script,
            julia_project=self.julia_project,
            julia_sysimage=self.julia_sysimage,
            trailing=["--worker"],
        )
        self._emit_status("Initializing BEAT Engine")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self.solver_script.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_julia_process_env(self.julia_threads, self.julia_project),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Julia executable was not found. Install Julia or set HORNLAB_BEAT_JULIA."
            ) from exc

        self._stderr_thread = threading.Thread(target=self._collect_stderr, daemon=True)
        self._stderr_thread.start()

        for event in self._read_events():
            event_type = str(event.get("type", ""))
            if event_type == "ready":
                self._emit_status("BEAT Engine ready")
                return
            if event_type == "failed":
                raise RuntimeError(
                    friendly_julia_error(
                        str(event.get("error", "BEAT Engine solver failed during startup.")),
                        julia_project=self.julia_project,
                    )
                )
        raise RuntimeError(self._process_error("Warm BEAT Engine solver ended before startup completed."))

    def _iter_events_for_submission(self) -> Iterator[dict]:
        try:
            for event in self._read_events():
                yield event
                if str(event.get("type", "")) in {"completed", "cancelled", "failed"}:
                    return
            raise RuntimeError(self._process_error("Warm BEAT Engine solver ended before job completion."))
        finally:
            self._status_callback = None
            if self._lock.locked():
                self._lock.release()

    def _read_events(self) -> Iterator[dict]:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                yield {"type": "status", "message": text}
                continue
            if isinstance(event, dict):
                yield event
        exit_code = process.wait()
        self._process = None
        if exit_code != 0:
            raise RuntimeError(self._process_error(f"Warm BEAT Engine solver exited with code {exit_code}."))

    def _collect_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.strip()
            if text:
                self._stderr_lines.append(text)
                self._emit_status(text)

    def _process_error(self, fallback: str) -> str:
        detail = "\n".join(self._stderr_lines[-10:])
        message = f"{fallback}\n{detail}" if detail else fallback
        return friendly_julia_error(
            message,
            julia_project=self.julia_project,
            detection_text="\n".join(self._stderr_lines),
        )

    def _emit_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)


def get_worker(
    *,
    julia_executable: str,
    solver_script: Path,
    julia_threads: str | int,
    julia_project: Path | None,
    julia_sysimage: Path | None = None,
) -> BeatWorkerProcess:
    resolved_threads = _resolve_julia_threads(julia_threads)
    key = (
        julia_executable,
        str(solver_script.resolve()),
        "" if julia_project is None else str(Path(julia_project).resolve()),
        "" if julia_sysimage is None else str(Path(julia_sysimage).resolve()),
        resolved_threads,
    )
    with _WORKERS_LOCK:
        worker = _WORKERS.get(key)
        if worker is None:
            worker = BeatWorkerProcess(
                julia_executable=julia_executable,
                solver_script=solver_script,
                julia_threads=resolved_threads,
                julia_project=julia_project,
                julia_sysimage=julia_sysimage,
            )
            _WORKERS[key] = worker
        return worker


def shutdown_workers() -> None:
    with _WORKERS_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        worker.terminate()


class BeatSolveSession:
    """One staged solve request against a (usually persistent) Julia worker.

    ``events()`` yields the raw solver events past initialization; the caller
    consumes ``result`` events until ``completed``/``cancelled``. Abandoning
    the iterator mid-solve requests cooperative cancellation and, for a
    persistent worker, terminates it rather than handing a mid-job process to
    the next request.
    """

    def __init__(
        self,
        request_payload: dict[str, Any],
        *,
        julia_executable: str,
        beat_backend: str,
        solver_script: Path = DEFAULT_SOLVER_SCRIPT,
        julia_project: Path | None = None,
        julia_threads: str | int = "auto",
        julia_sysimage: Path | None = None,
        persistent_worker: bool = True,
        status_callback: StatusCallback | None = None,
    ):
        self._status_callback = status_callback
        self._temp_dir = tempfile.TemporaryDirectory(prefix="hornlab-beat-")
        job_dir = Path(self._temp_dir.name)
        self._cancel_path = job_dir / "cancel"
        payload = dict(request_payload)
        payload["cancel_path"] = str(self._cancel_path)
        payload["beat_engine_backend"] = beat_backend
        request_path = job_dir / "request.json"
        request_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

        self.julia_project = julia_project if julia_project is not None else default_project(beat_backend)
        self.beat_backend = beat_backend
        self._worker: BeatWorkerProcess | None = None
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._events: Iterator[dict] | None = None
        self.metadata: dict[str, Any] | None = None
        self._finished = False

        if persistent_worker:
            self._worker = get_worker(
                julia_executable=julia_executable,
                solver_script=solver_script,
                julia_threads=julia_threads,
                julia_project=self.julia_project,
                julia_sysimage=julia_sysimage,
            )
            self._events = self._worker.submit(request_path, status_callback=self._emit_status)
        else:
            self._emit_status("Initializing BEAT Engine")
            try:
                self._process = subprocess.Popen(
                    _julia_command(
                        julia_executable,
                        solver_script,
                        julia_project=self.julia_project,
                        julia_sysimage=julia_sysimage,
                        trailing=["--request", str(request_path)],
                    ),
                    cwd=str(job_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=_julia_process_env(julia_threads, self.julia_project),
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Julia executable was not found. Install Julia or set HORNLAB_BEAT_JULIA."
                ) from exc
            threading.Thread(target=self._collect_stderr, daemon=True).start()
            self._events = self._iter_process_events()

        self._consume_until_initialized()

    def events(self) -> Iterator[dict]:
        assert self._events is not None
        try:
            for event in self._events:
                event_type = str(event.get("type", ""))
                if event_type == "status":
                    self._emit_status(str(event.get("message", "")))
                    continue
                if event_type == "failed":
                    self._finished = True
                    if self._worker is not None:
                        # A failed accelerator job may leave the runtime or
                        # allocator unhealthy; never reuse that worker.
                        self._worker.terminate()
                    raise RuntimeError(
                        friendly_julia_error(
                            str(event.get("error", "BEAT Engine solver failed.")),
                            julia_project=self.julia_project,
                            beat_backend=self.beat_backend,
                        )
                    )
                if event_type in {"completed", "cancelled"}:
                    self._finished = True
                    yield event
                    return
                yield event
        finally:
            self.close()

    def request_cancel(self) -> None:
        try:
            self._cancel_path.write_text("cancel", encoding="utf-8")
        except OSError:
            pass

    def close(self) -> None:
        events, self._events = self._events, None
        if events is not None:
            close = getattr(events, "close", None)
            if close is not None:
                close()
        if not self._finished:
            # Abandoned mid-solve: ask the solver to stop, and retire a
            # persistent worker instead of handing it over mid-job.
            self.request_cancel()
            if self._worker is not None:
                self._worker.terminate()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        try:
            self._temp_dir.cleanup()
        except OSError:
            pass

    def _consume_until_initialized(self) -> None:
        assert self._events is not None
        for event in self._events:
            event_type = str(event.get("type", ""))
            if event_type == "status":
                self._emit_status(str(event.get("message", "")))
                continue
            if event_type == "initialized":
                self.metadata = event
                return
            if event_type == "failed":
                self._finished = True
                if self._worker is not None:
                    self._worker.terminate()
                raise RuntimeError(
                    friendly_julia_error(
                        str(event.get("error", "BEAT Engine solver failed.")),
                        julia_project=self.julia_project,
                        beat_backend=self.beat_backend,
                    )
                )
            if event_type in {"completed", "cancelled"}:
                raise RuntimeError(f"BEAT Engine solver ended before initialization: {event_type}")
        raise RuntimeError("BEAT Engine solver ended before initialization.")

    def _iter_process_events(self) -> Iterator[dict]:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                self._emit_status(text)
                continue
            if isinstance(event, dict):
                yield event
        exit_code = process.wait()
        if exit_code != 0 and not self._finished:
            detail = "\n".join(self._stderr_lines[-10:])
            message = f"BEAT Engine solver exited with code {exit_code}."
            raise RuntimeError(
                friendly_julia_error(
                    f"{message}\n{detail}" if detail else message,
                    julia_project=self.julia_project,
                    beat_backend=self.beat_backend,
                    detection_text="\n".join(self._stderr_lines),
                )
            )

    def _collect_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.strip()
            if text:
                self._stderr_lines.append(text)
                self._emit_status(text)

    def _emit_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)


def friendly_julia_error(
    message: str,
    *,
    julia_project: Path | None,
    beat_backend: str | None = None,
    detection_text: str | None = None,
) -> str:
    """Turn a raw Julia dependency failure into an actionable instruction."""

    if julia_project is None:
        return message
    text = f"{detection_text or message}\n{message}".lower()
    markers = (
        "argumenterror: package",
        "not found in current path",
        "run `import pkg; pkg.add",
        "could not load project",
        "failed to precompile",
        "loading.jl",
        "require(into::module",
        "require(uuidkey::base.pkgid",
        "package cuda",
        "using cuda",
        "package amdgpu",
        "using amdgpu",
        "package metal",
        "using metal",
    )
    if not any(marker in text for marker in markers):
        return message
    project_path = Path(julia_project)
    label = {
        "cpu": "BEAT Engine (CPU)",
        "cuda": "BEAT Engine (Nvidia CUDA)",
        "rocm": "BEAT Engine (AMD ROCm)",
        "metal": "BEAT Engine (Apple Metal)",
    }.get(beat_backend or "", "the selected BEAT Engine backend")
    return (
        f"BEAT Engine could not load the Julia dependencies for {label}.\n\n"
        "The Julia environment has probably not been instantiated yet. Run:\n\n"
        f'julia --project={project_path} -e "using Pkg; Pkg.instantiate()"\n\n'
        f"Julia reported:\n{message}"
    )
