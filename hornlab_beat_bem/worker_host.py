"""The process that owns a Julia worker and outlives every client of it.

``python -m hornlab_beat_bem.worker_host --key <path>`` starts one. It is
spawned detached, it holds the Julia child on the same inherited-pipe protocol
the in-process worker has always used, and it serves that protocol to clients
over a socket. The Julia side is untouched: ``BeatEngineDriver.jl`` still
reads JSON lines from its stdin, which is exactly why this indirection exists
rather than a socket listener in Julia -- ``julia/src`` is a verbatim vendored
copy, and a transport change there would have to be re-applied on every sync.

What the extra process buys is the one thing pipes cannot give: the Julia
runtime's lifetime stops being the app's. A restart adopts the running worker
and skips the whole cold start, which on Metal is ~15 s of GPUCompiler work
that no disk cache can recover (GPUCompiler has none; see the cold-start
section of the README).

Three behaviours are worth reading the code for:

*Ownership.* One job runs at a time, exactly as the in-process worker's lock
enforced. A second client is queued rather than refused, because refusing
would turn a benign overlap -- an app's warm-up and its first solve -- into an
error the caller has no way to retry sensibly.

*Abandonment.* A client that vanishes mid-solve leaves Julia solving into a
pipe nobody reads. The host notices the dead connection, writes the request's
cancel file, retires the Julia child and starts a fresh one. Without that, an
adopting client would inherit a worker mid-job and read another job's events.

*Idleness.* With no connected client and no running job for the idle timeout,
the host removes its registry entry and exits. The timeout is generous by
design: it is the window in which an app restart is free, and its cost is one
idle Julia process.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import worker_registry as registry
from .worker_registry import KeyFile, receive_frame, send_frame


class WorkerHost:
    """Serves one Julia worker to a succession of clients."""

    def __init__(
        self,
        *,
        key: dict[str, Any],
        directory: Path,
        idle_timeout: float,
        token: str | None = None,
    ):
        self.key = key
        self.identifier = registry.key_id(key)
        self.directory = directory
        self.idle_timeout = idle_timeout
        self.token = token or registry.new_token()

        self._server: socket.socket | None = None
        self._endpoint: registry.Endpoint | None = None
        self._engine = self._build_engine()
        self._job_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._connections = 0
        self._job_running = False
        self._last_activity = time.monotonic()
        self._stopping = threading.Event()
        # Distinct from ``_stopping``, which every path that *asks* for a stop
        # sets. Only the accept loop performs the teardown, and it needs a
        # flag nobody else has already set or it would skip its own cleanup
        # and leave the registry entry behind.
        self._shut_down = threading.Event()
        self._warm_thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def _build_engine(self):
        from .worker import BeatWorkerProcess

        project = self.key["julia_project"]
        sysimage = self.key["julia_sysimage"]
        return BeatWorkerProcess(
            julia_executable=self.key["julia_executable"],
            solver_script=Path(self.key["solver_script"]),
            julia_threads=self.key["julia_threads"],
            julia_project=Path(project) if project else None,
            julia_sysimage=Path(sysimage) if sysimage else None,
        )

    def bind(self) -> registry.Endpoint:
        """Listen, and publish the address. Failure here means another host won.

        Binding before anything slow is deliberate: the spawning client polls
        for the key file, and a host that wrote it only after Julia had loaded
        would make every cold start look like a spawn timeout.
        """

        endpoint = registry.endpoint_for(self.identifier, self.directory)
        self._server = endpoint.listen()
        self._endpoint = endpoint
        registry.write_key_file(
            KeyFile(
                identifier=self.identifier,
                key=self.key,
                pid=os.getpid(),
                token=self.token,
                endpoint=endpoint,
            ),
            self.directory,
        )
        return endpoint

    def serve(self) -> None:
        assert self._server is not None, "bind() first"
        self._install_signal_handlers()
        self._start_background_warm()
        monitor = threading.Thread(target=self._idle_monitor, daemon=True)
        monitor.start()
        self._server.settimeout(0.5)
        try:
            while not self._stopping.is_set():
                try:
                    connection, _ = self._server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                thread = threading.Thread(
                    target=self._serve_connection, args=(connection,), daemon=True
                )
                thread.start()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shut_down.is_set():
            return
        self._shut_down.set()
        self._stopping.set()
        # The registry entry goes first: a client that reads it after this
        # point would find a socket nothing answers, which is a slower way to
        # learn the same thing.
        registry.remove_key_file(self.identifier, self.directory)
        if self._endpoint is not None:
            self._endpoint.cleanup()
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()
        self._engine.terminate()

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
            self._stopping.set()

        for name in ("SIGTERM", "SIGINT"):
            value = getattr(signal, name, None)
            if value is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(value, handler)

    # -- warmth ------------------------------------------------------------

    def _start_background_warm(self) -> None:
        """Boot Julia without waiting to be asked.

        This is what makes a cancelled solve cheap. Retiring a worker mid-job
        is unavoidable -- the runtime may be in any state -- but the cost of
        the retirement used to be paid by the *next* solve. Here it is paid by
        nobody: the replacement compiles while the user is deciding what to do
        next.
        """

        existing = self._warm_thread
        if existing is not None and existing.is_alive():
            return

        def run() -> None:
            try:
                self._engine.ensure_started()
            except Exception as exc:  # noqa: BLE001 - reported at the next op
                _log(f"background warm-up failed: {exc}")

        thread = threading.Thread(target=run, daemon=True)
        self._warm_thread = thread
        thread.start()

    # -- idleness ----------------------------------------------------------

    def _touch(self) -> None:
        with self._state_lock:
            self._last_activity = time.monotonic()

    def _idle_for(self) -> float:
        with self._state_lock:
            if self._connections > 0 or self._job_running:
                return 0.0
            return time.monotonic() - self._last_activity

    def _idle_monitor(self) -> None:
        interval = min(1.0, max(0.05, self.idle_timeout / 20.0))
        while not self._stopping.is_set():
            time.sleep(interval)
            if self._idle_for() >= self.idle_timeout:
                _log(f"idle for {self.idle_timeout:.0f}s; exiting")
                self._stopping.set()
                # Unblock the accept loop, which owns the shutdown.
                with contextlib.suppress(OSError):
                    if self._server is not None:
                        self._server.close()
                return

    # -- connections -------------------------------------------------------

    def _serve_connection(self, connection: socket.socket) -> None:
        connection.settimeout(None)
        greeted = False
        try:
            while not self._stopping.is_set():
                try:
                    message = receive_frame(connection)
                except (OSError, RuntimeError, ValueError):
                    return
                if message is None:
                    return
                operation = str(message.get("op", ""))
                if not greeted:
                    if operation != "hello":
                        send_frame(
                            connection,
                            {"type": "hello_refused", "reason": "hello must come first"},
                        )
                        return
                    if not self._greet(connection, message):
                        return
                    greeted = True
                    continue
                if not self._dispatch(connection, operation, message):
                    return
        except (OSError, BrokenPipeError):
            return
        finally:
            if greeted:
                with self._state_lock:
                    self._connections = max(0, self._connections - 1)
                    self._last_activity = time.monotonic()
            with contextlib.suppress(OSError):
                connection.close()

    def _greet(self, connection: socket.socket, message: dict[str, Any]) -> bool:
        """Verify identity before a client may drive the worker.

        Every field is checked even though the key id already hashes them: the
        id is a file name chosen for brevity, and a mismatch here is the
        difference between adopting the right runtime and adopting one built
        against a different package version.
        """

        reasons = []
        if int(message.get("protocol", -1)) != registry.PROTOCOL_VERSION:
            reasons.append(
                f"protocol {message.get('protocol')} != {registry.PROTOCOL_VERSION}"
            )
        if str(message.get("token", "")) != self.token:
            reasons.append("token mismatch")
        if str(message.get("key_id", "")) != self.identifier:
            reasons.append("key id mismatch")
        if message.get("key") != self.key:
            reasons.append("worker key mismatch")
        if reasons:
            with contextlib.suppress(OSError):
                send_frame(
                    connection,
                    {"type": "hello_refused", "reason": "; ".join(reasons)},
                )
            return False
        with self._state_lock:
            self._connections += 1
            self._last_activity = time.monotonic()
        send_frame(
            connection,
            {
                "type": "hello_ok",
                "host_pid": os.getpid(),
                "engine_pid": self._engine.pid,
                "protocol": registry.PROTOCOL_VERSION,
                "idle_timeout_s": self.idle_timeout,
            },
        )
        return True

    def _dispatch(
        self, connection: socket.socket, operation: str, message: dict[str, Any]
    ) -> bool:
        if operation == "ping":
            send_frame(connection, {"type": "pong", "host_pid": os.getpid()})
            self._touch()
            return True
        if operation == "ensure_started":
            self._ensure_started(connection)
            return True
        if operation == "submit":
            return self._submit(connection, message)
        if operation == "retire":
            self._retire()
            send_frame(connection, {"type": "retired"})
            return True
        if operation == "shutdown":
            send_frame(connection, {"type": "shutdown_ok"})
            self._stopping.set()
            with contextlib.suppress(OSError):
                if self._server is not None:
                    self._server.close()
            return False
        send_frame(connection, {"type": "failed", "error": f"unknown operation {operation!r}"})
        return True

    def _ensure_started(self, connection: socket.socket) -> None:
        def status(text: str) -> None:
            with contextlib.suppress(OSError):
                send_frame(connection, {"type": "status", "message": text})

        with self._job_lock:
            self._begin_job()
            try:
                self._engine.ensure_started(status_callback=status)
            except Exception as exc:  # noqa: BLE001 - the client renders it
                send_frame(connection, {"type": "failed", "error": str(exc)})
            else:
                send_frame(
                    connection,
                    {
                        "type": "ready",
                        "host_pid": os.getpid(),
                        "engine_pid": self._engine.pid,
                    },
                )
            finally:
                self._end_job()

    def _submit(self, connection: socket.socket, message: dict[str, Any]) -> bool:
        request_path = Path(str(message.get("request", "")))
        if not request_path.exists():
            send_frame(
                connection,
                {"type": "failed", "error": f"request file is missing: {request_path}"},
            )
            return True
        # The request file is the authority on where the cancel flag lives --
        # it is the same field the Julia driver reads. The message carries it
        # only as a fallback for a caller that staged the request itself.
        cancel_path = str(message.get("cancel_path", "") or "")
        try:
            staged = json.loads(request_path.read_text(encoding="utf-8"))
            cancel_path = str(staged.get("cancel_path") or cancel_path)
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

        def status(text: str) -> None:
            with contextlib.suppress(OSError):
                send_frame(connection, {"type": "status", "message": text})

        with self._job_lock:
            self._begin_job()
            events = None
            try:
                events = self._engine.submit(request_path, status_callback=status)
                for event in events:
                    send_frame(connection, event)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The client is gone. Julia is still solving; stop it and
                # replace it, or the next adopter reads this job's events.
                _log("client vanished mid-solve; cancelling and retiring the worker")
                _request_cancel(cancel_path)
                self._close_events(events)
                self._retire()
                return False
            except Exception as exc:  # noqa: BLE001 - the client renders it
                with contextlib.suppress(OSError):
                    send_frame(connection, {"type": "failed", "error": str(exc)})
                self._close_events(events)
                self._retire()
            finally:
                self._end_job()
        return True

    @staticmethod
    def _close_events(events: Any) -> None:
        if events is None:
            return
        close = getattr(events, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()

    def _retire(self) -> None:
        self._engine.terminate()
        if not self._stopping.is_set():
            self._start_background_warm()

    def _begin_job(self) -> None:
        with self._state_lock:
            self._job_running = True
            self._last_activity = time.monotonic()

    def _end_job(self) -> None:
        with self._state_lock:
            self._job_running = False
            self._last_activity = time.monotonic()


def _request_cancel(cancel_path: str) -> None:
    if not cancel_path:
        return
    with contextlib.suppress(OSError):
        Path(cancel_path).write_text("cancel", encoding="utf-8")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hornlab_beat_bem.worker_host")
    parser.add_argument("--key", required=True, help="path to a JSON file holding the worker key")
    parser.add_argument("--dir", default=None, help="registry directory (defaults to the standard one)")
    parser.add_argument("--idle-timeout", type=float, default=None)
    parser.add_argument("--token", default=None)
    arguments = parser.parse_args(argv)

    key = json.loads(Path(arguments.key).read_text(encoding="utf-8"))
    directory = Path(arguments.dir) if arguments.dir else registry.worker_dir()
    idle_timeout = (
        arguments.idle_timeout if arguments.idle_timeout is not None else registry.idle_timeout_s()
    )
    host = WorkerHost(
        key=key, directory=directory, idle_timeout=idle_timeout, token=arguments.token
    )
    try:
        endpoint = host.bind()
    except OSError as exc:
        # Another host won the race for this key. That is a normal outcome,
        # not a failure: the client that spawned us will find the winner.
        _log(f"could not bind for key {host.identifier}: {exc}")
        return 0
    _log(f"serving key {host.identifier} on {endpoint.as_dict()} (idle {idle_timeout:.0f}s)")
    host.serve()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
