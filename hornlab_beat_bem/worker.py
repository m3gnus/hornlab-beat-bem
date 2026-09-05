"""Julia process management for the vendored BEAT Engine solver.

Ported from boundary-lab's ``blab/solvers/beat_engine_backend.py`` (GPL-3.0):
a persistent stdin/stdout JSON-lines worker per (executable, project, threads,
sysimage) key, a one-shot subprocess fallback, file-based cooperative
cancellation, and warm-up. The asset-staging layer is gone -- callers here
always hand the solver an on-disk mesh path directly.

Since 0.3.2 that worker is, by default, **not a child of this process**. The
pipe worker below is still exactly what talks to Julia, but it normally runs
inside a detached host process (``worker_host``) that a later application
launch can adopt over a socket (``worker_client``), because the cold start it
saves cannot be cached: GPUCompiler has no disk cache, so a Metal worker pays
~15 s of kernel compilation once per *process* no matter how warm the depot
is. ``HORNLAB_BEAT_PERSISTENT_HOST=0`` restores the child-process behaviour.

The lifetime contract for an embedding application:

``shutdown_workers()``
    stops the host, and with it the Julia runtime. The next launch pays the
    full cold start.
``detach_workers()`` (or ``shutdown_workers(detach=True)``)
    lets go of the host and leaves it running. An application's quit hook
    wants this one; the host retires itself after its idle timeout.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import platform
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from . import worker_registry as _registry
from .config import BEAT_CUDA, BEAT_METAL, BEAT_ROCM
from .runtime import (
    DEFAULT_SOLVER_SCRIPT,
    default_project,
    package_fingerprint,
    package_version,
)
from .submission import Submission, SubmissionSlot, SubmissionStream
from .worker_client import HostedBeatWorker, keyed_environment

StatusCallback = Callable[[str], None]

_WORKERS_LOCK = threading.Lock()
_WORKERS: dict[str, BeatWorkerProcess | HostedBeatWorker] = {}


@functools.lru_cache(maxsize=1)
def _performance_core_count() -> int:
    """Cores worth giving the dense factorization, not every core there is.

    On an asymmetric Apple Silicon part ``os.cpu_count()`` counts the
    efficiency cores too, and Julia's threads are pinned nowhere, so a
    thread lands on one and the whole factorization waits for it. Measured
    on an 8P+2E M1 Max, asro68 quarter, one sweep: 10 threads 7.69 s,
    9 threads 7.29 s, 8 threads 6.41 s. Targeting the performance cores is
    worth 1.20x for one number.

    ``hw.perflevel0`` is the performance level and exists only on asymmetric
    parts; everywhere else this falls through to the full count, which is
    then already the right answer.
    """
    total = os.cpu_count() or 1
    if platform.system() != "Darwin":
        return total
    try:
        text = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        count = int(text)
    except (OSError, ValueError, subprocess.SubprocessError):
        return total
    if count < 1:
        return total
    return min(count, total)


def _resolve_julia_threads(julia_threads: str | int = "auto") -> str:
    if isinstance(julia_threads, int):
        return str(max(1, julia_threads))
    text = str(julia_threads or "auto").strip().lower()
    if text == "auto":
        return str(_performance_core_count())
    try:
        return str(max(1, int(text)))
    except ValueError:
        return str(_performance_core_count())


def _julia_process_env(julia_threads: str | int, julia_project: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = _resolve_julia_threads(julia_threads)
    # BLAB_METAL_PIPELINE is deliberately not set here any more. This function
    # runs when the worker starts, which is before any mesh has been read, and
    # whether overlapping the sweep pays is a property of the mesh: it is a loss
    # on the small symmetry-reduced meshes this package usually serves and a
    # ~1.5x win on a full model. A process-wide value could only ever be right
    # for one of those, so the choice moved into the solver, which knows the dof
    # count -- see `metal_pipeline_requested` in julia/BeatEngineDriver.jl and
    # the README's "Sweep threads and sweep pipelining".
    #
    # An explicit BLAB_METAL_PIPELINE in the caller's environment still reaches
    # the solver through the copy above and still wins, in both directions.
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
    """One persistent Julia worker; one submission at a time.

    "One at a time" is enforced by a :class:`~.submission.SubmissionSlot`
    rather than a plain lock, because every caller that ends a job here is an
    unwind path and unwind paths run out of order. The slot records which
    submission holds it, so a release or a retirement arriving from a request
    that has already finished is refused instead of being applied to whoever
    holds the worker now.
    """

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
        self._lock = SubmissionSlot()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._status_callback: StatusCallback | None = None

    @property
    def pid(self) -> int | None:
        """The Julia process id, or None when no runtime is loaded.

        Reported to clients so an adoption can be verified against the
        *runtime*, not merely against the host that owns it -- a host whose
        Julia child was retired and replaced is warm again, but it is not the
        same compilation.
        """

        process = self._process
        if process is None or process.poll() is not None:
            return None
        return process.pid

    def submit(
        self,
        request_path: Path,
        *,
        status_callback: StatusCallback | None = None,
    ) -> SubmissionStream:
        submission = self._lock.acquire()
        try:
            # Inside the try, not between it and the acquire: everything that
            # happens after the slot is taken has to be covered by the unwind
            # that gives it back.
            self._status_callback = status_callback
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
            return SubmissionStream(
                self._lock, submission, self._iter_events_for_submission(submission)
            )
        except BaseException:
            self._status_callback = None
            self._lock.release(submission)
            raise

    def ensure_started(self, *, status_callback: StatusCallback | None = None) -> None:
        with self._lock.held():
            previous = self._status_callback
            self._status_callback = status_callback
            try:
                self._ensure_started()
            finally:
                self._status_callback = previous

    def terminate(self) -> bool:
        """Stop the Julia child, whoever is using it.

        This is the *unconditional* retirement: the application shutting down,
        or the host replacing a runtime it owns outright. It deliberately does
        not touch the submission slot. It used to release the slot whenever it
        found it held, which is how an earlier request came to free a later
        one's submission -- a plain lock cannot tell the two apart, and by the
        time an abandoned session reached here it had already let go. A caller
        that is ending its *own* submission wants ``retire`` below.

        An in-flight submission is not left stranded by the silence: the
        stream it is reading ends with the process, and the owner releases the
        slot on its way out.

        Returns whether the runtime was retired, which in process is always:
        the child is this object's to kill and there is no second client to
        decline for. The hosted worker's answer is the one that varies.
        """

        self._kill_process()
        return True

    def retire(self, submission: Submission | None) -> bool:
        """Stop the Julia child on behalf of the submission that holds it.

        Refused, and a no-op, once this submission no longer owns the worker.
        That case is real rather than defensive: a stream that ended because
        the runtime died has already released the slot, and by then the
        process this would kill belongs to somebody else.
        """

        return self._lock.retire(submission, self._kill_process)

    def release_submission(self, submission: Submission | None) -> bool:
        """Hand the slot back, if this submission still holds it."""

        return self._lock.release(submission)

    def refuse_submissions(self) -> None:
        """Take no further work, and unblock whoever is waiting for some.

        Ownership means a submission can only be released by its owner, which
        is the property that makes a lost owner unrecoverable: nobody else may
        free the slot, and ``acquire`` waits without a deadline because a
        queued solve is meant to wait for the one ahead of it. This is the way
        out, and it belongs to the process rather than to a request --
        ``shutdown_workers`` calls it on a worker it is dropping, so a thread
        parked on a solve that will now never end is told so instead of
        waiting for a runtime that is being stopped underneath it.
        """

        self._lock.close()

    def _kill_process(self) -> None:
        """Stop the Julia child and forget it, without touching the slot."""

        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

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
                # Drop the corpse before reporting. Leaving it in place used
                # to make the *next* ``ensure_started`` see a process that had
                # not been polled yet, take the fast path above, and report a
                # worker that was already dead -- harmless while a failed
                # start-up ended the process, and not harmless at all now that
                # a host retries.
                self._kill_process()
                raise RuntimeError(
                    friendly_julia_error(
                        str(event.get("error", "BEAT Engine solver failed during startup.")),
                        julia_project=self.julia_project,
                    )
                )
        message = self._process_error("Warm BEAT Engine solver ended before startup completed.")
        self._kill_process()
        raise RuntimeError(message)

    def _iter_events_for_submission(self, submission: Submission) -> Iterator[dict]:
        try:
            for event in self._read_events():
                yield event
                if str(event.get("type", "")) in {"completed", "cancelled", "failed"}:
                    return
            raise RuntimeError(self._process_error("Warm BEAT Engine solver ended before job completion."))
        finally:
            self._status_callback = None
            self._lock.release(submission)

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


def worker_key(
    *,
    julia_executable: str,
    solver_script: Path,
    julia_threads: str | int,
    julia_project: Path | None,
    julia_sysimage: Path | None = None,
) -> dict[str, Any]:
    """The full identity of the worker a request needs.

    Beyond the four paths and the thread count the in-process registry always
    used, this adds the installed package version, a content fingerprint of
    the wrapper and the vendored engine, and the Julia-relevant part of the
    environment. All three matter only because a worker now outlives the
    process that started it: adopting a runtime built from different code, or
    with a different depot, produces correct-looking numbers from the wrong
    solver and has no symptom at all. The fingerprint is what actually carries
    that weight -- see ``runtime.package_fingerprint`` for why the version
    number cannot, in a repository consumers pin by SHA.
    """

    return _registry.worker_key(
        julia_executable=julia_executable,
        solver_script=solver_script,
        julia_project=julia_project,
        julia_sysimage=julia_sysimage,
        julia_threads=_resolve_julia_threads(julia_threads),
        package_version=package_version(),
        package_fingerprint=package_fingerprint(julia_project, julia_sysimage),
        environment=keyed_environment(),
    )


def get_worker(
    *,
    julia_executable: str,
    solver_script: Path,
    julia_threads: str | int,
    julia_project: Path | None,
    julia_sysimage: Path | None = None,
) -> BeatWorkerProcess | HostedBeatWorker:
    """The worker for this configuration, adopting a running one where possible.

    The returned object is a persistent host client by default and a plain
    child process under ``HORNLAB_BEAT_PERSISTENT_HOST=0``. Both expose the
    same ``ensure_started`` / ``submit`` / ``terminate`` surface, which is all
    ``BeatSolveSession`` uses.
    """

    key = worker_key(
        julia_executable=julia_executable,
        solver_script=solver_script,
        julia_threads=julia_threads,
        julia_project=julia_project,
        julia_sysimage=julia_sysimage,
    )
    identifier = _registry.key_id(key)
    hosted = _registry.persistent_host_enabled()
    cache_key = f"{'host' if hosted else 'child'}:{identifier}"
    with _WORKERS_LOCK:
        worker = _WORKERS.get(cache_key)
        if worker is None:
            if hosted:
                worker = HostedBeatWorker(key)
            else:
                worker = BeatWorkerProcess(
                    julia_executable=julia_executable,
                    solver_script=solver_script,
                    julia_threads=_resolve_julia_threads(julia_threads),
                    julia_project=julia_project,
                    julia_sysimage=julia_sysimage,
                )
            _WORKERS[cache_key] = worker
        return worker


def shutdown_workers(*, detach: bool = False) -> None:
    """Release this process's workers.

    ``detach=True`` is the one an application's quit hook wants: it closes the
    connections and leaves the host processes running, so the next launch
    adopts a warm Julia runtime instead of compiling one. The hosts retire
    themselves after ``HORNLAB_BEAT_WORKER_IDLE_S`` (30 minutes by default),
    so nothing accumulates.

    The default still stops everything, because that is what a script, a test
    and an explicit "stop the solver" request all mean. An in-process worker
    (``HORNLAB_BEAT_PERSISTENT_HOST=0``) has nothing to detach from and is
    terminated either way -- it dies with this process regardless.

    Either way the worker's submission slot is closed first. These worker
    objects are being dropped from the cache, so nothing may be admitted to
    one afterwards -- and a thread already queued behind a request that has
    stopped making progress has, until it is told, no way out at all: only the
    owner may free the slot, and the owner is the request being abandoned.
    Closing the slot makes that thread raise where it would otherwise wait for
    a worker this process no longer holds.
    """

    with _WORKERS_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        with contextlib.suppress(Exception):
            worker.refuse_submissions()
    # Every worker is stopped even if one of them refuses to die, and the
    # first failure is re-raised afterwards. They have already been dropped
    # from the cache, so a stop abandoned half way through leaves runtimes
    # nothing can reach.
    failure: BaseException | None = None
    for worker in workers:
        try:
            if detach and isinstance(worker, HostedBeatWorker):
                worker.detach()
            elif isinstance(worker, HostedBeatWorker):
                worker.shutdown()
            else:
                worker.terminate()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


def detach_workers() -> None:
    """Let go of every persistent worker without stopping it.

    The sibling of ``shutdown_workers`` named in the application contract: a
    quit hook calls this so the Julia runtime survives the application and the
    next launch skips the cold start.
    """

    shutdown_workers(detach=True)


class BeatSolveSession:
    """One staged solve request against a (usually persistent) Julia worker.

    ``events()`` yields the raw solver events past initialization; the caller
    consumes ``result`` events until ``completed``/``cancelled``. Abandoning
    the iterator mid-solve requests cooperative cancellation and, for a
    persistent worker, terminates it rather than handing a mid-job process to
    the next request.

    Construction is where a solve can fail before the caller holds anything to
    ``close``: staging the request, the worker look-up, the process launch and
    the initial handshake all happen here. Each of those runs inside
    ``__init__``'s unwind (``_abort_start``), because a session that raised is
    unreachable -- nobody can close it, and whatever it had taken (the staged
    directory, a one-shot child, the worker's submission slot) would be held
    until the garbage collector happened to run, which for the slot means the
    *next* request blocks on a solve that no longer exists.

    What the session owns of the worker is one :class:`~.submission.Submission`
    and nothing else. Both ends of the lifetime follow from that: retirement
    happens *before* the event stream is closed, because closing it is what
    gives the submission back, and a session that has already given it back
    retires nothing. That ordering is the difference between a worker this
    session condemned and one the next request is already solving on.
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
        # Every attribute the unwind touches exists before anything that can
        # fail runs, so ``_abort_start`` never has to guess how far the
        # constructor got.
        self._status_callback = status_callback
        self.beat_backend = beat_backend
        self.julia_project = julia_project
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._cancel_path: Path | None = None
        self._worker: BeatWorkerProcess | HostedBeatWorker | None = None
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._events: Iterator[dict] | None = None
        self.metadata: dict[str, Any] | None = None
        self._finished = False
        #: True once a request is actually in flight. Until then a failure is
        #: this request's alone and must leave the worker to the next one.
        self._submitted = False
        #: This session's claim on the persistent worker, and the only thing
        #: that entitles it to release or retire one. ``None`` means the
        #: worker is not ours -- never taken, or already given back.
        self._submission: Submission | None = None
        self._retire_worker = False
        self._closed = False

        try:
            if self.julia_project is None:
                self.julia_project = default_project(beat_backend)
            self._temp_dir = tempfile.TemporaryDirectory(prefix="hornlab-beat-")
            job_dir = Path(self._temp_dir.name)
            self._cancel_path = job_dir / "cancel"
            payload = dict(request_payload)
            payload["cancel_path"] = str(self._cancel_path)
            payload["beat_engine_backend"] = beat_backend
            request_path = job_dir / "request.json"
            request_path.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )

            if persistent_worker:
                self._worker = get_worker(
                    julia_executable=julia_executable,
                    solver_script=solver_script,
                    julia_threads=julia_threads,
                    julia_project=self.julia_project,
                    julia_sysimage=julia_sysimage,
                )
                # ``submit`` holds the worker's submission slot from here until
                # the stream is closed, and only from here: it releases the
                # slot itself if it raises. This is the instruction that makes
                # the worker ours to cancel and retire, and not one earlier --
                # and the submission it hands back is the proof we hold when
                # we say so.
                #
                # The submission is stored *first*, and the stream second,
                # because an asynchronous exception can land between the two
                # stores and the unwind reads them in that order: with only
                # the stream stored, ``_abort_start`` would close it -- giving
                # the slot back -- without ever cancelling the job that is
                # still running on the far side. With only the submission
                # stored it cancels, retires and releases, which is the
                # correct handling of a request that is in flight and whose
                # events nobody will read. The one window neither store covers
                # -- the instant between ``submit`` returning and the first
                # assignment -- is the stream's finalizer's, best-effort.
                stream = self._worker.submit(
                    request_path, status_callback=self._emit_status
                )
                self._submission = stream.submission
                self._events = stream
                self._submitted = True
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
                self._submitted = True
                threading.Thread(target=self._collect_stderr, daemon=True).start()
                self._events = self._iter_process_events()

            self._consume_until_initialized()
        except BaseException:
            self._abort_start()
            raise

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
                    # A failed accelerator job may leave the runtime or
                    # allocator unhealthy; never reuse that worker. The
                    # retirement itself happens in ``close`` below, once the
                    # event stream has been closed -- see ``_close_events``.
                    self._retire_worker = True
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
        if self._cancel_path is None:
            return
        try:
            self._cancel_path.write_text("cancel", encoding="utf-8")
        except OSError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._finished:
            # Abandoned mid-solve: ask the solver to stop, and retire a
            # persistent worker instead of handing it over mid-job.
            self.request_cancel()
            self._retire_worker = True
        # Retirement first, and the stream second. Closing the stream is what
        # releases the submission, so the other order would publish the worker
        # to whoever is waiting and only then kill it.
        #
        # Every step runs even when an earlier one raises, and the first
        # failure is re-raised once they all have. Retirement is the step that
        # can genuinely fail -- a Julia child that survives both a SIGTERM and
        # a SIGKILL for two seconds raises ``TimeoutExpired`` out of the kill
        # -- and letting that skip the release would strand the submission for
        # the lifetime of the process: this session is already ``_closed``, so
        # nothing will call it again, and only the holder may free the slot.
        # A runtime that would not stop is still worth reporting; it is simply
        # not worth a leaked worker as well.
        events, self._events = self._events, None
        failure: BaseException | None = None
        for step in (
            self._retire_submission,
            functools.partial(self._close_events, events),
            self._release_submission,
            self._terminate_process,
            self._cleanup_temp_dir,
        ):
            try:
                step()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure

    def _abort_start(self) -> None:
        """Give back everything a failed ``__init__`` had taken.

        The caller of a constructor that raised has no object to ``close``, so
        the unwind happens here, and it has to preserve the exception on its
        way out: a secondary failure while releasing (a generator whose
        ``finally`` throws, a child that will not die) is suppressed, because
        the error being propagated is the one that explains the failure.

        The worker is retired only when this session actually had it. A
        failure before ``submit`` returned -- serialising the request, the
        status callback, the worker look-up, a runtime that never started --
        belongs to this request alone, and terminating on it would take down a
        warm worker that the next request, or a concurrent one, is entitled to.
        """

        self._closed = True
        if (self._submitted or self._submission is not None) and not self._finished:
            # The worker may still be mid-job on our request: same contract as
            # an abandoned solve. Holding the submission counts as having
            # submitted even when the flag says otherwise -- an asynchronous
            # exception can land between those two stores, and the half that
            # matters here is the one that says a job is running.
            self.request_cancel()
            self._retire_worker = True
        with contextlib.suppress(BaseException):
            self._retire_submission()
        events, self._events = self._events, None
        with contextlib.suppress(BaseException):
            self._close_events(events)
        with contextlib.suppress(BaseException):
            self._release_submission()
        with contextlib.suppress(BaseException):
            self._terminate_process()
        with contextlib.suppress(BaseException):
            self._cleanup_temp_dir()

    def _retire_submission(self) -> None:
        """Retire the worker, while this session still owns its submission.

        Both halves of that sentence carry weight. *While*: the submission is
        still held here, because the stream has not been closed yet, so no
        waiting request can be admitted between the decision and the kill.
        *Owns*: a session whose stream already ended -- a runtime that died
        under it, say -- released the submission when the stream did, and the
        slot refuses the retirement rather than applying it to whoever took
        the worker next.
        """

        if not self._retire_worker:
            return
        self._retire_worker = False
        if self._worker is None:
            return
        self._worker.retire(self._submission)

    def _release_submission(self) -> None:
        """Give the submission back, if closing the stream has not already.

        Normally it has -- ``SubmissionStream.close`` releases whether or not
        its generator ever ran. This is the backstop for the paths where that
        close does not happen or does not finish: an unwind that suppressed an
        exception out of it, or a session that never reached a stream at all.
        Releasing twice is safe by construction, because the slot only honours
        a release from the submission that still holds it.
        """

        submission, self._submission = self._submission, None
        if self._worker is not None:
            self._worker.release_submission(submission)

    @staticmethod
    def _close_events(events: Iterator[dict] | None) -> None:
        """Close the event stream, which is what releases the submission.

        It runs *after* ``_retire_submission`` for that reason. The order used
        to be the other way, to keep a late generator close from releasing a
        lock that had been re-taken by the next request -- a hazard the
        submission slot removes outright, since a release from a submission
        that no longer holds the slot is now a no-op rather than a hand-over.
        """

        if events is None:
            return
        close = getattr(events, "close", None)
        if close is None:
            return
        with contextlib.suppress(Exception):
            close()

    def _terminate_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)

    def _cleanup_temp_dir(self) -> None:
        temp_dir, self._temp_dir = self._temp_dir, None
        if temp_dir is None:
            return
        with contextlib.suppress(OSError):
            temp_dir.cleanup()

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
                # Retired by the unwind, after the stream has been closed.
                self._retire_worker = True
                raise RuntimeError(
                    friendly_julia_error(
                        str(event.get("error", "BEAT Engine solver failed.")),
                        julia_project=self.julia_project,
                        beat_backend=self.beat_backend,
                    )
                )
            if event_type in {"completed", "cancelled"}:
                # The submission is over -- there is no job to cancel and no
                # reason to distrust the runtime, so the worker stays warm for
                # the next request; only this session's own state is released.
                self._finished = True
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
