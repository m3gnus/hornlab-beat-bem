"""Suite-wide isolation for the persistent worker registry.

Workers now outlive the process that started them, which is exactly the
property that makes an unisolated test suite dangerous in both directions: a
test could adopt (or shut down) a developer's warm worker, and a leaked test
host could sit in the real registry until its idle timeout. Every test in this
suite therefore gets a private registry directory, and anything still running
in it at the end is killed.

The directory deliberately sits under a short prefix: a Unix socket path has
about a hundred bytes to spend, and pytest's own temporary directories on
macOS spend most of them before the file name starts. Long paths are handled
-- ``endpoint_for`` falls back to loopback TCP -- but silently taking the
fallback everywhere would leave the Unix transport untested on the platform
where it matters most.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from hornlab_beat_bem import worker_registry as registry
from hornlab_beat_bem.worker import shutdown_workers
from hornlab_beat_bem.worker_client import find_live_hosts

_MANAGED_ENV = (
    registry.WORKER_DIR_ENV_VAR,
    registry.IDLE_TIMEOUT_ENV_VAR,
    registry.SPAWN_TIMEOUT_ENV_VAR,
    registry.TRANSPORT_ENV_VAR,
)


def _short_temporary_root() -> str:
    system = Path("/tmp")
    if os.name == "posix" and system.is_dir():
        return str(system)
    return tempfile.gettempdir()


@pytest.fixture(scope="session", autouse=True)
def isolated_worker_registry():
    previous = {name: os.environ.get(name) for name in _MANAGED_ENV}
    directory = Path(tempfile.mkdtemp(prefix="hlb-t-", dir=_short_temporary_root()))
    os.environ[registry.WORKER_DIR_ENV_VAR] = str(directory)
    # A safety net rather than a behaviour under test: anything that escapes
    # the teardown below still retires itself within two minutes.
    os.environ.setdefault(registry.IDLE_TIMEOUT_ENV_VAR, "120")
    try:
        yield directory
    finally:
        shutdown_workers()
        for record in find_live_hosts(directory):
            registry.terminate_pid(record.pid)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(directory, ignore_errors=True)
