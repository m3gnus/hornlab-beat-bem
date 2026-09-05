"""Client side of the persistent worker: find one, or start one, then drive it.

``HostedBeatWorker`` is a drop-in for ``BeatWorkerProcess`` -- same three
methods, same event stream -- with one difference that is the entire point:
its Julia runtime belongs to a host process that neither started with this
application nor ends with it. Calling ``get_worker`` in a freshly launched
app usually *adopts* a runtime that is already warm, which on Metal is worth
the ~15 s of kernel compilation that no cache can recover.

Adoption is only safe if identity is exact, so it is checked twice: once
against the registry record before connecting, and once by the host itself
during the handshake. The failure this guards against is quiet -- a worker
built from a different package version answers every request perfectly and
returns the previous version's numbers.

The crash cases are the interesting part of this file:

*The host died between writing its record and now.* The record is present,
the pid may even be alive (recycled), and the connect fails or the handshake
is refused. Both are treated as a stale record: clear it and spawn.

*The socket file outlives the host.* A Unix socket is a filesystem entry; an
abrupt kill leaves it behind and ``connect`` returns ECONNREFUSED. Same
handling, and the removal happens under the spawn lock so a client that is
mid-spawn does not have its fresh socket deleted by a client that is mid-scan.

*Two applications start at once.* Look-up, cleanup and spawn all happen inside
a cross-process file lock, so the second application's look-up runs after the
first application's host has published itself, and finds it.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from . import worker_registry as registry
from .worker_registry import receive_frame, send_frame

StatusCallback = Callable[[str], None]

#: Environment variables that change what the Julia runtime *does*, and so
#: must match before a worker may be adopted. A worker started with a
#: different depot, load path or BLAB tuning is a different solver wearing the
#: same paths; adopting it would silently answer with the old configuration.
#: Everything else (PATH, terminal settings, the caller's own variables) is
#: deliberately excluded -- keying on the full environment would make adoption
#: fail for reasons that have nothing to do with the solve.
_KEYED_ENV_NAMES = ("JULIA_DEPOT_PATH", "JULIA_LOAD_PATH", "JULIA_PROJECT")
_KEYED_ENV_PREFIXES = ("BLAB_",)


def keyed_environment() -> dict[str, str]:
    keyed = {
        name: value
        for name, value in os.environ.items()
        if name in _KEYED_ENV_NAMES or name.startswith(_KEYED_ENV_PREFIXES)
    }
    return dict(sorted(keyed.items()))


class WorkerHandshakeError(RuntimeError):
    """A live host refused this client's identity."""


class HostedBeatWorker:
    """A Julia worker owned by a detached host process."""

    def __init__(self, key: dict[str, Any], *, directory: Path | None = None):
        self.key = key
        self.identifier = registry.key_id(key)
        self.directory = directory if directory is not None else registry.worker_dir()
        self._lock = threading.Lock()
        self._connection: socket.socket | None = None
        self.host_pid: int | None = None
        #: The Julia process id behind the host, as of the last handshake or
        #: ``ensure_started``. An adoption that preserves this preserved the
        #: compilation; one that only preserved ``host_pid`` did not.
        self.engine_pid: int | None = None

    # -- public surface, mirroring BeatWorkerProcess ------------------------

    def ensure_started(self, *, status_callback: StatusCallback | None = None) -> None:
        with self._lock:
            connection = self._connect()
            send_frame(connection, {"op": "ensure_started"})
            for event in self._read_until(connection, {"ready", "failed"}):
                event_type = str(event.get("type", ""))
                if event_type == "status" and status_callback is not None:
                    status_callback(str(event.get("message", "")))
                elif event_type == "ready":
                    self.engine_pid = event.get("engine_pid")
                elif event_type == "failed":
                    raise RuntimeError(
                        str(event.get("error", "BEAT Engine worker failed during startup."))
                    )

    def submit(
        self,
        request_path: Path,
        *,
        status_callback: StatusCallback | None = None,
    ) -> Iterator[dict]:
        self._lock.acquire()
        try:
            connection = self._connect()
            # The host reads the cancel path out of the staged request; this
            # is only the fallback for a request that does not carry one.
            send_frame(
                connection,
                {
                    "op": "submit",
                    "request": str(request_path),
                    "cancel_path": str(Path(request_path).parent / "cancel"),
                },
            )
        except BaseException:
            self._lock.release()
            raise
        return self._iter_submission(connection, status_callback)

    def terminate(self) -> None:
        """Retire the Julia child; the host survives and warms a replacement.

        This is what an abandoned or failed solve calls, and the change from
        the in-process worker is that it no longer costs the *next* solve a
        cold start: the host starts a fresh runtime immediately, so the
        compilation happens while nobody is waiting for it.

        It deliberately takes no lock. The failure path calls it from *inside*
        the submission generator, with the lock still held -- the in-process
        worker had the same contract, and taking the lock here would deadlock
        exactly when a solve has already gone wrong.

        Either half of the sequence below is sufficient on its own: the host
        retires on an explicit ``retire``, and it also retires when a
        connection dies with a job running. Both run because the abandoned
        case cannot be distinguished from the failed one at this level -- when
        the host is mid-send, the ``retire`` frame waits behind a full socket
        buffer and only the close is heard; when the host is idle, only the
        frame is.
        """

        self._say_and_close({"op": "retire"}, "retired")
        if self._lock.locked():
            with contextlib.suppress(RuntimeError):
                self._lock.release()

    def shutdown(self) -> None:
        """Stop the host process itself, ending the persistence."""

        self._say_and_close({"op": "shutdown"}, "shutdown_ok")

    def detach(self) -> None:
        """Let go of the host without stopping it. The app may now quit."""

        self._close_connection()

    def ping(self) -> dict[str, Any]:
        with self._lock:
            return self._request({"op": "ping"}, {"pong"}, timeout=10.0)

    # -- plumbing ----------------------------------------------------------

    def _say_and_close(self, message: dict[str, Any], terminal: str) -> None:
        connection = self._connection
        if connection is None:
            return
        deadline = time.monotonic() + 2.0
        try:
            send_frame(connection, message)
            connection.settimeout(2.0)
            while time.monotonic() < deadline:
                event = receive_frame(connection)
                if event is None or str(event.get("type", "")) == terminal:
                    break
        except (OSError, RuntimeError, ValueError):
            pass
        finally:
            self._close_connection()

    def _request(
        self, message: dict[str, Any], terminal: set[str], *, timeout: float
    ) -> dict[str, Any]:
        connection = self._connect()
        connection.settimeout(timeout)
        try:
            send_frame(connection, message)
            for event in self._read_until(connection, terminal):
                if str(event.get("type", "")) in terminal:
                    return event
        finally:
            with contextlib.suppress(OSError):
                connection.settimeout(None)
        raise RuntimeError(f"BEAT worker host closed before answering {message['op']!r}")

    def _read_until(
        self, connection: socket.socket, terminal: set[str]
    ) -> Iterator[dict[str, Any]]:
        while True:
            event = receive_frame(connection)
            if event is None:
                self._close_connection()
                raise RuntimeError("BEAT worker host closed the connection unexpectedly")
            yield event
            if str(event.get("type", "")) in terminal:
                return

    def _iter_submission(
        self, connection: socket.socket, status_callback: StatusCallback | None
    ) -> Iterator[dict]:
        try:
            while True:
                event = receive_frame(connection)
                if event is None:
                    self._close_connection()
                    raise RuntimeError(
                        "BEAT worker host ended the connection before job completion."
                    )
                event_type = str(event.get("type", ""))
                if event_type == "status" and status_callback is not None:
                    status_callback(str(event.get("message", "")))
                yield event
                if event_type in {"completed", "cancelled", "failed"}:
                    return
        finally:
            if self._lock.locked():
                with contextlib.suppress(RuntimeError):
                    self._lock.release()

    def _close_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            with contextlib.suppress(OSError):
                connection.close()

    def _connect(self) -> socket.socket:
        """Adopt a running host, or start one; never return a broken adoption.

        A registry record is a claim, not a fact -- the host may have been
        killed since it was written, leaving a socket file nobody answers, or
        it may be wedged with its pid still alive. Both look identical here,
        and both are resolved the same way: the failed record is passed to
        ``_spawn_host``, which clears exactly that record under the spawn lock
        rather than clearing whatever it happens to find (which by then may be
        a healthy host another client just published).
        """

        if self._connection is not None:
            return self._connection
        record = self._live_record()
        connection: socket.socket | None = None
        if record is not None:
            try:
                connection = self._handshake(record)
            except (OSError, WorkerHandshakeError, RuntimeError, ValueError):
                connection = None
        if connection is None:
            record = self._spawn_host(replacing=record)
            connection = self._handshake(record)
        self._connection = connection
        self.host_pid = record.pid
        return connection

    def _live_record(self) -> registry.KeyFile | None:
        """The registry entry for this key, if it still describes a real host."""

        record = registry.read_key_file(self.identifier, self.directory)
        if record is None:
            return None
        if record.key != self.key:
            # Written by a different shape of this package, or corrupt. It is
            # not ours to drive, and the key id is a hash of the key, so it
            # cannot legitimately live at this path.
            return None
        if not registry.pid_alive(record.pid):
            return None
        return record

    def _handshake(self, record: registry.KeyFile) -> socket.socket:
        connection = record.endpoint.connect(timeout=10.0)
        send_frame(
            connection,
            {
                "op": "hello",
                "protocol": registry.PROTOCOL_VERSION,
                "key_id": self.identifier,
                "key": self.key,
                "token": record.token,
            },
        )
        reply = receive_frame(connection)
        connection.settimeout(None)
        if reply is None or str(reply.get("type", "")) != "hello_ok":
            with contextlib.suppress(OSError):
                connection.close()
            reason = "the host closed the connection" if reply is None else reply.get("reason")
            raise WorkerHandshakeError(f"BEAT worker host refused this client: {reason}")
        self.engine_pid = reply.get("engine_pid")
        return connection

    def _spawn_host(self, *, replacing: registry.KeyFile | None = None) -> registry.KeyFile:
        """Start a host for this key, or discover the one that beat us to it.

        Everything from the second look-up to the moment the new host has
        published itself happens under the spawn lock. Releasing earlier would
        put a competing client's look-up in the window where no record exists
        yet, and it would start a second host for the same key.

        ``replacing`` is the record that was just found unusable, if any. The
        look-up under the lock accepts any record that is not that one: while
        we were waiting for the lock, another client may have replaced the
        broken host with a working one, and starting a third would be wrong.
        """

        lock_path = registry.spawn_lock_path(self.identifier, self.directory)
        with registry.SpawnLock(lock_path, timeout=registry.spawn_timeout_s() + 30.0):
            record = self._live_record()
            if record is not None and not _same_record(record, replacing):
                return record
            self._clear_stale_entry()
            process = self._launch()
            deadline = time.monotonic() + registry.spawn_timeout_s()
            while time.monotonic() < deadline:
                record = self._live_record()
                if record is not None:
                    return record
                if process.poll() is not None:
                    raise RuntimeError(self._launch_failure(process.returncode))
                time.sleep(0.02)
            raise RuntimeError(
                "The BEAT worker host did not become reachable within "
                f"{registry.spawn_timeout_s():.0f}s. See {registry.log_path(self.identifier, self.directory)}."
            )

    def _clear_stale_entry(self) -> None:
        """Retire the record at this key, whatever state it is in.

        Only reached under the spawn lock, and only once the caller has
        established that the record cannot be used -- so a pid that is still
        alive is a wedged host holding an endpoint nobody can reach, and
        leaving it would make every future spawn fail to bind.
        """

        record = registry.read_key_file(self.identifier, self.directory)
        if record is not None:
            if registry.pid_alive(record.pid) and record.pid != os.getpid():
                registry.terminate_pid(record.pid)
            record.endpoint.cleanup()
        registry.remove_key_file(self.identifier, self.directory)
        socket_path = self.directory / f"{self.identifier}.sock"
        if socket_path.exists():
            with contextlib.suppress(OSError):
                socket_path.unlink()

    def _launch(self) -> subprocess.Popen:
        key_path = self.directory / f"{self.identifier}.key.json"
        key_path.write_text(json.dumps(self.key, sort_keys=True), encoding="utf-8")
        command = [
            registry.python_executable(),
            "-m",
            "hornlab_beat_bem.worker_host",
            "--key",
            str(key_path),
            "--dir",
            str(self.directory),
            "--idle-timeout",
            str(registry.idle_timeout_s()),
        ]
        log = registry.log_path(self.identifier, self.directory)
        handle = open(log, "wb")
        creation_flags = 0
        start_new_session = False
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: no console, and no
            # Ctrl-C from the launcher's console group, which is how a
            # packaged app's shutdown would otherwise take the host with it.
            creation_flags = 0x00000008 | 0x00000200
        else:
            # setsid: the host leaves the caller's process group and session,
            # so a terminal's SIGINT/SIGHUP and a launcher that kills its own
            # group both leave it running. This is the line that makes the
            # worker outlive the app.
            start_new_session = True
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.directory),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=self._host_environment(),
                start_new_session=start_new_session,
                creationflags=creation_flags,
                close_fds=True,
            )
        finally:
            handle.close()
        # A host is still this process's child even in its own session, so
        # nothing reaps it when it retires and ``pid_alive`` would keep
        # answering yes for a zombie -- an adoption check that can never fail
        # is worse than no check. One parked thread per host settles it, and
        # if this process exits first, init inherits and reaps.
        threading.Thread(target=process.wait, daemon=True).start()
        return process

    def _host_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        # The host builds the Julia environment itself; give it the same
        # registry settings the client resolved so a spawned host and an
        # adopting client never disagree about where the registry lives.
        environment[registry.WORKER_DIR_ENV_VAR] = str(self.directory)
        return environment

    def _launch_failure(self, returncode: int | None) -> str:
        log = registry.log_path(self.identifier, self.directory)
        detail = ""
        with contextlib.suppress(OSError):
            detail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:])
        message = (
            f"The BEAT worker host exited with code {returncode} before it "
            "became reachable."
        )
        return f"{message}\n{detail}" if detail else message


def _same_record(record: registry.KeyFile, other: registry.KeyFile | None) -> bool:
    if other is None:
        return False
    return (
        record.pid == other.pid
        and record.token == other.token
        and record.endpoint.as_dict() == other.endpoint.as_dict()
    )


def find_live_hosts(directory: Path | str | None = None) -> list[registry.KeyFile]:
    """Every host this user currently has running. Diagnostics, and tests."""

    directory = Path(directory) if directory is not None else registry.worker_dir()
    records = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".key.json"):
            continue
        record = registry.read_key_file(path.stem, directory)
        if record is not None and registry.pid_alive(record.pid):
            records.append(record)
    return records


def temporary_registry() -> tempfile.TemporaryDirectory:
    """A throwaway registry directory, for tests and one-off scripts."""

    return tempfile.TemporaryDirectory(prefix="hornlab-beat-registry-")
