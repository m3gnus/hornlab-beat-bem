"""Naming, addressing and liveness for Julia workers that outlive their client.

A BEAT worker is expensive to start and impossible to cache: Julia's JIT work
is recovered by the engine bundles, but GPUCompiler has no disk cache, so a
Metal worker pays ~15 s of kernel compilation once **per process**, however
warm the depot is. The only way to stop paying it on every app restart is to
stop ending the process with the app -- which means the worker has to be
findable by a Python process that did not start it.

This module is that lookup. It answers three questions and nothing else:

``worker_key``
    Which worker would satisfy this request? Two clients share a worker only
    when the executable, the script, the project, the sysimage, the thread
    count, the package version, a content fingerprint of the wrapper and the
    vendored engine, the Julia-relevant environment and the wire protocol all
    match. The last four are the ones that were not needed while a worker was
    a child process: within one process they cannot differ, and across an
    adoption they can. Adopting across any of them returns yesterday's solver
    answering today's request, with no symptom at all.

``Endpoint``
    Where does it listen? A Unix domain socket where one is available and the
    path fits in ``sun_path``, and a loopback TCP port otherwise -- which is
    both the Windows answer and the fallback for a home directory too deep to
    address. The endpoint round-trips through the key file, so a client never
    needs to know which was chosen.

``KeyFile``
    Is it still there? The registry directory holds one JSON file per key,
    naming the host pid, the endpoint and a per-host token. Every field is
    checked before adoption, because each of them fails differently: a dead
    pid is an orphan record, a live pid with a refused connect is a host that
    died between binding and now, and a mismatched key is a file this package
    did not write in this shape.

Nothing here talks to Julia, and nothing here starts a process; the host is in
``worker_host`` and the client in ``worker_client``.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import socket
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Bumped whenever the client/host wire protocol changes shape. It is part of
#: the key, so a new client never adopts a worker that speaks the old one --
#: an important property for an editable install, where both versions of this
#: file exist on the same machine within one working day.
PROTOCOL_VERSION = 1

#: Directory holding one key file (and one Unix socket) per live worker.
WORKER_DIR_ENV_VAR = "HORNLAB_BEAT_WORKER_DIR"
#: Seconds a host stays alive with no connected client and no running job.
IDLE_TIMEOUT_ENV_VAR = "HORNLAB_BEAT_WORKER_IDLE_S"
#: Seconds a client waits for a freshly spawned host to become connectable.
SPAWN_TIMEOUT_ENV_VAR = "HORNLAB_BEAT_WORKER_SPAWN_TIMEOUT_S"
#: "0" disables the out-of-process host entirely (in-process pipe worker).
PERSISTENT_HOST_ENV_VAR = "HORNLAB_BEAT_PERSISTENT_HOST"
#: "unix" or "tcp"; forces a transport that would not otherwise be chosen.
TRANSPORT_ENV_VAR = "HORNLAB_BEAT_WORKER_TRANSPORT"

DEFAULT_IDLE_TIMEOUT_S = 1800.0
DEFAULT_SPAWN_TIMEOUT_S = 60.0

#: ``sun_path`` is 104 bytes on macOS and 108 on Linux, including the NUL.
#: Exceeding it is an ``OSError`` at bind time, not a truncation, and the
#: paths that do it are ordinary ones -- a deep home directory plus a
#: per-session scratchpad. Rather than special-case the failure, a path this
#: long simply selects the loopback transport, which has no such limit.
_MAX_UNIX_PATH = 100


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0.0 else default


def idle_timeout_s() -> float:
    return _env_float(IDLE_TIMEOUT_ENV_VAR, DEFAULT_IDLE_TIMEOUT_S)


def spawn_timeout_s() -> float:
    return _env_float(SPAWN_TIMEOUT_ENV_VAR, DEFAULT_SPAWN_TIMEOUT_S)


def persistent_host_enabled() -> bool:
    return os.environ.get(PERSISTENT_HOST_ENV_VAR, "1").strip() != "0"


def _default_worker_dir(platform_name: str | None = None) -> Path:
    """Return a stable per-user registry root for the current platform."""

    platform_name = os.name if platform_name is None else platform_name
    if platform_name == "nt":
        # Windows has no getuid(). LOCALAPPDATA is per-user and survives an
        # application restart, unlike the old pid-suffixed temp directory.
        # Do not use APPDATA: it can roam to other machines with the worker
        # token and logs. A home-relative local directory is the normal
        # fallback when LOCALAPPDATA is absent.
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            root = Path(local_app_data)
        else:
            try:
                root = Path.home() / "AppData" / "Local"
            except (OSError, RuntimeError) as exc:
                # A system temp directory may be shared across accounts. The
                # registry contains the host token, so ambiguous ownership is
                # a reason to refuse persistence rather than guess.
                raise RuntimeError(
                    "Cannot determine a private per-user BEAT worker directory "
                    "on Windows. Set HORNLAB_BEAT_WORKER_DIR to a local "
                    "directory readable only by this user."
                ) from exc
        return root / "HornLab" / "BEAT" / "workers"

    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if runtime_dir and Path(runtime_dir).is_dir():
        return Path(runtime_dir) / "hornlab-beat"
    return Path(tempfile.gettempdir()) / f"hornlab-beat-{os.getuid()}"


def worker_dir() -> Path:
    """The per-user registry directory, created with owner-only permissions.

    ``XDG_RUNTIME_DIR`` is preferred on Linux because it is already per-user,
    already 0700 and already cleaned at logout. Other POSIX systems use a
    uid-suffixed directory under the system temporary directory. Windows uses
    the user's local application-data directory and refuses to choose a shared
    fallback when no user profile is available. Every selected default is
    stable across application processes, which is required for worker adoption.
    """

    override = os.environ.get(WORKER_DIR_ENV_VAR, "").strip()
    if override:
        directory = Path(override)
    else:
        directory = _default_worker_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
    return directory


def worker_key(
    *,
    julia_executable: str,
    solver_script: Path,
    julia_project: Path | None,
    julia_sysimage: Path | None,
    julia_threads: str,
    package_version: str | None,
    package_fingerprint: str = "",
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The identity a worker must match before a client may adopt it."""

    def resolved(value: Path | str | None) -> str:
        if value is None:
            return ""
        try:
            return str(Path(value).resolve())
        except OSError:
            return str(value)

    return {
        "protocol": PROTOCOL_VERSION,
        "julia_executable": resolved(julia_executable),
        "solver_script": resolved(solver_script),
        "julia_project": resolved(julia_project),
        "julia_sysimage": resolved(julia_sysimage),
        "julia_threads": str(julia_threads),
        "package_version": package_version or "",
        "package_fingerprint": package_fingerprint or "",
        "environment": dict(sorted((environment or {}).items())),
    }


def key_id(key: dict[str, Any]) -> str:
    """A short, stable file name for a key.

    Truncated to 16 hex characters because it becomes part of a Unix socket
    path, where every byte competes with ``sun_path``. It names a file in a
    directory only this user can read; it is not a security boundary, and the
    token in the key file is.
    """

    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------


class Endpoint:
    """Where a host listens. Serialisable, because the client reads it back."""

    kind = ""

    def as_dict(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def listen(self) -> socket.socket:  # pragma: no cover - overridden
        raise NotImplementedError

    def connect(self, timeout: float) -> socket.socket:  # pragma: no cover
        raise NotImplementedError

    def cleanup(self) -> None:
        """Remove any filesystem residue this endpoint owns."""

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> Endpoint:
        kind = str(raw.get("kind", ""))
        if kind == "unix":
            return UnixEndpoint(Path(str(raw["path"])))
        if kind == "tcp":
            return TcpEndpoint(int(raw["port"]), host=str(raw.get("host", "127.0.0.1")))
        raise ValueError(f"unknown worker endpoint kind: {kind!r}")


@dataclass
class UnixEndpoint(Endpoint):
    path: Path
    kind = "unix"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "unix", "path": str(self.path)}

    def listen(self) -> socket.socket:
        # A leftover socket file whose host is gone would make bind() fail with
        # EADDRINUSE forever; a live host holding it makes bind() fail too, and
        # that failure is the race guard, so only unlink what nothing answers.
        if self.path.exists() and not _unix_socket_answers(self.path):
            try:
                self.path.unlink()
            except OSError:
                pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.path))
        except OSError:
            server.close()
            raise
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        server.listen(16)
        return server

    def connect(self, timeout: float) -> socket.socket:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        try:
            client.connect(str(self.path))
        except OSError:
            client.close()
            raise
        return client

    def cleanup(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


@dataclass
class TcpEndpoint(Endpoint):
    port: int
    host: str = "127.0.0.1"
    kind = "tcp"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "tcp", "host": self.host, "port": int(self.port)}

    def listen(self) -> socket.socket:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Deliberately no SO_REUSEADDR: a second host binding the same port is
        # exactly the race this failure is meant to lose.
        try:
            server.bind((self.host, int(self.port)))
        except OSError:
            server.close()
            raise
        server.listen(16)
        self.port = server.getsockname()[1]
        return server

    def connect(self, timeout: float) -> socket.socket:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(timeout)
        try:
            client.connect((self.host, int(self.port)))
        except OSError:
            client.close()
            raise
        return client


def _unix_socket_answers(path: Path) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect(str(path))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _unix_available() -> bool:
    return hasattr(socket, "AF_UNIX") and os.name == "posix"


def endpoint_for(identifier: str, directory: Path | None = None) -> Endpoint:
    """Choose a transport for a key, honouring the test/Windows override."""

    directory = directory if directory is not None else worker_dir()
    forced = os.environ.get(TRANSPORT_ENV_VAR, "").strip().lower()
    if forced == "tcp":
        return TcpEndpoint(0)
    path = directory / f"{identifier}.sock"
    if forced == "unix" or (_unix_available() and len(str(path).encode()) <= _MAX_UNIX_PATH):
        if not _unix_available():
            raise RuntimeError("Unix domain sockets are not available on this platform")
        return UnixEndpoint(path)
    # Windows, or a socket path longer than sun_path: an ephemeral loopback
    # port, whose number the host records in the key file once bound.
    return TcpEndpoint(0)


# --------------------------------------------------------------------------
# Key files
# --------------------------------------------------------------------------


@dataclass
class KeyFile:
    identifier: str
    key: dict[str, Any]
    pid: int
    token: str
    endpoint: Endpoint
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_VERSION,
            "key_id": self.identifier,
            "key": self.key,
            "pid": int(self.pid),
            "token": self.token,
            "endpoint": self.endpoint.as_dict(),
            "started_at": float(self.started_at),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> KeyFile:
        return KeyFile(
            identifier=str(raw["key_id"]),
            key=dict(raw["key"]),
            pid=int(raw["pid"]),
            token=str(raw["token"]),
            endpoint=Endpoint.from_dict(dict(raw["endpoint"])),
            started_at=float(raw.get("started_at", 0.0)),
        )


def key_file_path(identifier: str, directory: Path | None = None) -> Path:
    directory = directory if directory is not None else worker_dir()
    return directory / f"{identifier}.json"


def log_path(identifier: str, directory: Path | None = None) -> Path:
    directory = directory if directory is not None else worker_dir()
    return directory / f"{identifier}.log"


def write_key_file(record: KeyFile, directory: Path | None = None) -> Path:
    """Publish a host's address, atomically and owner-readable only.

    Atomically because a client that reads a half-written file would see a
    JSON error and conclude the record is corrupt, then clean up a host that
    is perfectly healthy.
    """

    path = key_file_path(record.identifier, directory)
    temporary = path.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record.as_dict(), sort_keys=True, indent=2), encoding="utf-8"
    )
    if os.name == "posix":
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
    os.replace(temporary, path)
    return path


def read_key_file(identifier: str, directory: Path | None = None) -> KeyFile | None:
    path = key_file_path(identifier, directory)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return KeyFile.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return None


def remove_key_file(identifier: str, directory: Path | None = None) -> None:
    try:
        key_file_path(identifier, directory).unlink()
    except OSError:
        pass


def new_token() -> str:
    return secrets.token_hex(16)


# --------------------------------------------------------------------------
# Liveness
# --------------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    """Whether a process id still exists.

    Not ``os.kill(pid, 0)`` on Windows: there, ``os.kill`` with any signal
    other than the two console events calls ``TerminateProcess``, so the
    idiomatic POSIX liveness probe would kill the worker it is asking about.
    """

    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid)
        )
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user: alive, but not ours. Treated as alive so the
        # caller refuses rather than deletes.
        return True
    except OSError:
        return False
    return True


def terminate_pid(pid: int) -> None:
    """Best-effort stop for a host that must not be adopted and will not answer."""

    if not pid_alive(pid):
        return
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import signal

            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, 15)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Spawn lock
# --------------------------------------------------------------------------


class SpawnLock:
    """A cross-process lock held while a host is being started.

    Two applications launched together find no worker at the same instant and
    both spawn one. The loser's host then fails to bind and exits, which is
    correct but wasteful and leaves a confusing log; more importantly the
    loser has no way to learn that the winner's host is now the one to use
    without re-reading the registry under exclusion. So the whole
    look-then-spawn sequence happens inside this lock, and the loser's second
    look finds the winner.

    ``flock`` (POSIX) and ``msvcrt.locking`` (Windows) are both released by
    the kernel when the holder dies, which is the property a pid file cannot
    offer: a client killed mid-spawn must not wedge every later client.
    """

    def __init__(self, path: Path, timeout: float = 60.0):
        self._path = path
        self._timeout = timeout
        self._handle: Any = None

    def __enter__(self) -> SpawnLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "a+b")
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                self._acquire()
                return self
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for the BEAT worker spawn lock at {self._path}"
                    ) from exc
                time.sleep(0.02)

    def __exit__(self, *exc_info: object) -> None:
        try:
            self._release()
        finally:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def _acquire(self) -> None:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt

            # Lock a fixed byte, not "wherever append mode left the cursor":
            # the position after opening in append mode is not the same on
            # every platform or run, and two clients locking different bytes
            # would both succeed.
            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(self) -> None:
        if self._handle is None:
            return
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt

            try:
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            return
        import fcntl

        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


def spawn_lock_path(identifier: str, directory: Path | None = None) -> Path:
    directory = directory if directory is not None else worker_dir()
    return directory / f"{identifier}.lock"


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------
#
# Length-prefixed JSON rather than newline-delimited: solver events carry
# whole pressure blocks, a single frame runs to megabytes, and a reader that
# scans for a newline has to buffer the same bytes twice. The prefix also
# makes a truncated frame -- the signature of a host that died mid-solve --
# distinguishable from a slow one.

_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 512 * 1024 * 1024


def send_frame(connection: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection.sendall(_HEADER.pack(len(body)) + body)


def receive_frame(connection: socket.socket) -> dict[str, Any] | None:
    header = _receive_exactly(connection, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > _MAX_FRAME_BYTES:
        raise RuntimeError(f"BEAT worker frame of {length} bytes exceeds the limit")
    body = _receive_exactly(connection, length)
    if body is None:
        return None
    decoded = json.loads(body.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("BEAT worker frame was not a JSON object")
    return decoded


def _receive_exactly(connection: socket.socket, count: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = connection.recv(min(remaining, 1 << 20))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def python_executable() -> str:
    """The interpreter used to start a host.

    ``sys.executable`` is empty in a few embedded configurations; there is no
    sane fallback, so the caller is told rather than handed ``""``.
    """

    if not sys.executable:
        raise RuntimeError(
            "sys.executable is empty, so no persistent BEAT worker host can be "
            f"started; set {PERSISTENT_HOST_ENV_VAR}=0 to use an in-process worker"
        )
    return sys.executable
