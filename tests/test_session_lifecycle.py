"""What a ``BeatSolveSession`` owns, and what it gives back when it fails.

``test_worker_persistence.py`` covers the worker's own lifetime -- adoption,
retirement, re-warm. This file covers the seam above it: the constructor, which
stages a request directory, takes the worker's submission lock and waits for
the first protocol event, and which is the one place where a failure leaves the
caller nothing to close. Everything asserted here is therefore some form of the
same question -- after this failure, can the next solve still run?

Two claims are load-bearing and appear in almost every test:

* the staged directory is gone, deterministically, not whenever the garbage
  collector next runs;
* the worker is left exactly as the failure warrants -- retired when the
  session had a job on it, and *untouched* when the failure happened before
  the request was ever submitted, because that worker belongs to the next
  request as much as to this one.

The fake worker stands in for Julia, as it does next door: the lifecycle is
identical and a real cold start per test is not affordable.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from hornlab_beat_bem import worker_registry as registry
from hornlab_beat_bem.worker import (
    BeatSolveSession,
    BeatWorkerProcess,
    get_worker,
    shutdown_workers,
)
from hornlab_beat_bem.worker_client import find_live_hosts

FAKE_WORKER = Path(__file__).resolve().parent / "fake_julia_worker.py"

BASE_PAYLOAD = {
    "schema_version": 2,
    "config": {"tag_throat": 2},
    "frequencies_hz": [100.0, 200.0],
}


@pytest.fixture
def staging(monkeypatch, tmp_path):
    """A private root for the session's staged request directories.

    ``BeatSolveSession`` names them ``hornlab-beat-*`` under the default
    temporary directory; pointing that at an empty directory is what makes
    "nothing was left behind" a statement about this test and not about
    whatever else the machine is doing.
    """

    root = tmp_path / "staging"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def staged(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.name.startswith("hornlab-beat-"))


@pytest.fixture
def child_worker(monkeypatch, staging):
    """Sessions backed by an in-process child worker, not a detached host.

    The child worker is the same ``submit`` / ``terminate`` surface the hosted
    one implements, and it puts the submission lock and the Julia child in this
    process where a test can look at them directly. The hosted path gets its
    own test below.
    """

    monkeypatch.setenv(registry.PERSISTENT_HOST_ENV_VAR, "0")
    # The stand-in interpreter does not accept ``--project=``.
    monkeypatch.setattr("hornlab_beat_bem.worker.default_project", lambda backend: None)
    shutdown_workers()
    yield staging
    shutdown_workers()


def make_session(*, payload: dict | None = None, **kwargs) -> BeatSolveSession:
    arguments = {
        "julia_executable": sys.executable,
        "beat_backend": "cpu",
        "solver_script": FAKE_WORKER,
        "julia_project": None,
        "julia_threads": "1",
    }
    arguments.update(kwargs)
    return BeatSolveSession(dict(BASE_PAYLOAD, **(payload or {})), **arguments)


def current_worker() -> BeatWorkerProcess:
    worker = get_worker(
        julia_executable=sys.executable,
        solver_script=FAKE_WORKER,
        julia_threads="1",
        julia_project=None,
    )
    assert isinstance(worker, BeatWorkerProcess)
    return worker


def event_types(session: BeatSolveSession) -> list[str]:
    return [str(event.get("type")) for event in session.events()]


# --------------------------------------------------------------------------
# failures before the request is ever submitted
# --------------------------------------------------------------------------


def test_a_request_that_cannot_be_serialised_stages_nothing_and_starts_nothing(
    child_worker,
):
    """The first thing that can fail, and the earliest resource to leak.

    Serialisation happens after the directory exists and before any worker has
    been asked for anything, so the constructor is already holding something
    the caller will never see.
    """

    with pytest.raises(TypeError) as failure:
        make_session(payload={"config": object()})

    # The traceback is held on purpose, here and below: it references the
    # session that raised, and without that reference the interpreter would
    # collect the session at the end of the ``with`` block and its finalizer
    # would remove the directory. "Cleaned up whenever the collector gets to
    # it" is the behaviour under test, not the behaviour being asserted.
    assert failure.traceback
    assert staged(child_worker) == []
    from hornlab_beat_bem import worker as worker_module

    assert worker_module._WORKERS == {}, "no worker should have been touched"


def test_a_one_shot_launch_that_cannot_start_stages_nothing(child_worker):
    with pytest.raises(RuntimeError, match="Julia executable was not found") as failure:
        make_session(
            julia_executable=str(Path(child_worker) / "no-such-julia"),
            persistent_worker=False,
        )

    assert failure.traceback
    assert staged(child_worker) == []


def test_a_worker_that_cannot_start_stages_nothing_and_frees_the_lock(child_worker):
    """A start-up failure is not a submission, so nothing is left held.

    The submission lock is the resource that matters here: it is taken inside
    ``submit`` before the runtime is started, and a start-up that fails must
    hand it straight back or every later request queues behind a solve that
    never began.
    """

    with pytest.raises(RuntimeError, match="Julia executable was not found") as failure:
        make_session(julia_executable=str(Path(child_worker) / "no-such-julia"))

    assert failure.traceback
    assert staged(child_worker) == []
    worker = get_worker(
        julia_executable=str(Path(child_worker) / "no-such-julia"),
        solver_script=FAKE_WORKER,
        julia_threads="1",
        julia_project=None,
    )
    assert not worker._lock.locked()


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_a_status_callback_that_fails_at_submission_leaves_the_worker_warm(
    child_worker, error_type,
):
    """The failure that must *not* cost anyone their runtime.

    The callback belongs to this caller, and it throws before the request has
    reached the solver. Retiring the worker for it would throw away a warm
    Julia runtime -- the expensive thing this whole layer exists to keep -- for
    a fault that has nothing to do with the runtime, and would do it to
    whatever request came next as well.
    """

    worker = current_worker()
    worker.ensure_started()
    warm_pid = worker.pid
    assert warm_pid is not None

    def explode(message: str) -> None:
        if message == "Submitting solve request":
            raise error_type("status callback failed")

    with pytest.raises(error_type, match="status callback failed") as failure:
        make_session(status_callback=explode)

    assert failure.traceback
    assert staged(child_worker) == []
    assert not worker._lock.locked()
    assert worker.pid == warm_pid, "a pre-submission failure must not retire the worker"

    session = make_session()
    assert session.metadata is not None
    assert event_types(session) == ["result", "result", "completed"]
    assert current_worker().pid == warm_pid


# --------------------------------------------------------------------------
# failures during the initial protocol exchange
# --------------------------------------------------------------------------


def test_a_solver_that_completes_before_initialising_releases_the_lock(child_worker):
    """The half-initialized session the caller cannot close.

    The constructor raises while the submission generator is suspended on a
    yield, holding the lock. Nothing releases it except closing that generator,
    which only the constructor can still do -- so if it does not, the next
    request blocks forever on a solve that ended before it started.
    """

    worker = current_worker()
    worker.ensure_started()
    warm_pid = worker.pid

    with pytest.raises(RuntimeError, match="ended before initialization: completed") as failure:
        make_session(payload={"fake_complete_before_initialized": True})

    assert failure.traceback
    assert staged(child_worker) == []
    assert not worker._lock.locked(), "the submission lock was leaked"
    # The submission ended cleanly; there is no job to cancel and no reason to
    # distrust the runtime, so the warm one is still there.
    assert worker.pid == warm_pid

    session = make_session()
    assert event_types(session) == ["result", "result", "completed"]
    assert staged(child_worker) == []


def test_a_solver_that_fails_before_initialising_retires_the_worker(child_worker):
    """A failed job may have left the runtime unhealthy; the next one gets a new one."""

    worker = current_worker()
    worker.ensure_started()
    failed_pid = worker.pid
    assert failed_pid is not None

    with pytest.raises(RuntimeError, match="fake worker was told to fail this request") as failure:
        make_session(payload={"fake_fail_before_initialized": True})

    assert failure.traceback
    assert staged(child_worker) == []
    assert not worker._lock.locked()
    assert worker.pid is None, "the failed runtime must not be handed on"

    session = make_session()
    assert session.metadata is not None
    assert event_types(session) == ["result", "result", "completed"]
    assert current_worker().pid not in (None, failed_pid)


def test_a_failed_start_does_not_disturb_a_worker_it_never_used(child_worker):
    """Two configurations, one failed job, and the other must not notice.

    Retirement is targeted or it is destructive: with two runtimes in the
    process, a failure on one must leave the other exactly as warm as it found
    it, rather than clearing whatever the cleanup path can reach.
    """

    other = get_worker(
        julia_executable=sys.executable,
        solver_script=FAKE_WORKER,
        julia_threads="2",
        julia_project=None,
    )
    other.ensure_started()
    other_pid = other.pid
    assert other_pid is not None

    with pytest.raises(RuntimeError, match="fake worker was told to fail this request") as failure:
        make_session(payload={"fake_fail_before_initialized": True})

    assert failure.traceback
    assert other.pid == other_pid
    assert not other._lock.locked()
    assert staged(child_worker) == []


# --------------------------------------------------------------------------
# cancellation and abandonment at the session API
# --------------------------------------------------------------------------


def test_request_cancel_ends_the_stream_with_partial_results(child_worker, monkeypatch):
    """The explicit partial-result contract, driven through the session.

    A cancelled sweep keeps the results it already produced and ends on
    ``cancelled``. It is an orderly ending, not a fault, so the worker is
    handed on warm.
    """

    worker = current_worker()
    worker.ensure_started()
    warm_pid = worker.pid

    session = make_session(payload={"frequencies_hz": [100.0, 200.0, 300.0], "fake_wait_for_cancel": True})
    seen = []
    for event in session.events():
        seen.append(str(event.get("type")))
        if seen == ["result"]:
            session.request_cancel()

    assert seen[0] == "result"
    assert seen[-1] == "cancelled"
    assert len(seen) < 4, "cancellation must cut the sweep short"
    assert staged(child_worker) == []
    assert not worker._lock.locked()
    assert current_worker().pid == warm_pid


def test_abandoning_a_session_mid_solve_retires_the_worker_and_the_next_solve_runs(
    child_worker, monkeypatch
):
    """Closing mid-stream hands nothing half-solved to the next request."""

    monkeypatch.setenv("FAKE_BEAT_SOLVE_S", "0.05")
    worker = current_worker()
    worker.ensure_started()
    abandoned_pid = worker.pid
    assert abandoned_pid is not None

    session = make_session(payload={"frequencies_hz": [100.0, 200.0, 300.0]})
    stream = session.events()
    assert str(next(stream).get("type")) == "result"
    session.close()

    assert staged(child_worker) == []
    assert not worker._lock.locked()
    assert worker.pid is None
    # Closing twice is what a `finally` around an abandoned stream does.
    session.close()

    session = make_session()
    assert event_types(session) == ["result", "result", "completed"]
    assert current_worker().pid not in (None, abandoned_pid)


# --------------------------------------------------------------------------
# the same unwind through the detached host
# --------------------------------------------------------------------------


@pytest.fixture
def hosted_worker(monkeypatch, staging, isolated_worker_registry):
    directory = Path(tempfile.mkdtemp(prefix="s-", dir=isolated_worker_registry))
    monkeypatch.setenv(registry.WORKER_DIR_ENV_VAR, str(directory))
    monkeypatch.setattr("hornlab_beat_bem.worker.default_project", lambda backend: None)
    shutdown_workers()
    yield staging
    shutdown_workers()
    for record in find_live_hosts(directory):
        registry.terminate_pid(record.pid)


def test_a_failed_start_through_the_host_leaves_a_usable_host(hosted_worker):
    """The hosted worker releases its lock in the same place, over a socket.

    The client's submission lock and the host's one-job-at-a-time rule are
    separate things, and a constructor that raised has to let go of both --
    otherwise the next session in this process blocks locally and never even
    reaches the host that is sitting there, warm and idle.
    """

    with pytest.raises(RuntimeError, match="fake worker was told to fail this request") as failure:
        make_session(payload={"fake_fail_before_initialized": True})

    assert failure.traceback
    assert staged(hosted_worker) == []

    session = make_session()
    assert session.metadata is not None
    assert event_types(session) == ["result", "result", "completed"]
    assert staged(hosted_worker) == []
