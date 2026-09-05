"""Who owns a worker's submission slot, and who is allowed to end it.

``test_session_lifecycle.py`` asks whether a failed session gives back what it
took. This file asks the harder question next to it: when two requests meet at
one worker, can the *earlier* one reach into the *later* one? Both worker
implementations used to answer yes, in two different ways:

* ``terminate`` released the submission lock whenever it found it held --
  ``threading.Lock`` records no owner, so "held" meant "held by somebody", and
  the somebody was routinely the next request rather than the caller;
* retirement was not atomic. A session closing mid-solve released the lock
  (by closing its event stream) and only then killed the runtime, so a request
  admitted in between was handed a worker that the previous request was about
  to destroy.

A third question follows from the answer to those two. Ownership means only
the owner may give the slot back, so a request that ends without giving it
back parks every later one forever -- and the last section here is about the
way out: closing the slot, which admits nobody and wakes everybody.

Every handshake below is an ``Event`` or a wait on a state another thread has
reached (``SubmissionSlot.await_waiters``); the timeouts are deadlock guards
with a deliberately absurd value, and a test that fails on one has hung rather
than lost a race. The one duration in the file is inherited rather than
chosen: ``fake_wait_for_cancel`` in the stand-in worker gives up after
``FAKE_BEAT_CANCEL_WAIT_S`` so that a test which never cancels fails instead
of hanging, and the two-client test below widens that fallback to the same
guard for exactly that reason.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from hornlab_beat_bem import worker_registry as registry
from hornlab_beat_bem.worker import (
    BeatSolveSession,
    BeatWorkerProcess,
    get_worker,
    shutdown_workers,
)
from hornlab_beat_bem.worker_client import HostedBeatWorker, find_live_hosts

FAKE_WORKER = Path(__file__).resolve().parent / "fake_julia_worker.py"

#: A guard against a hang, not a margin against a slow machine. Every wait in
#: this file is on an event another thread sets; if one expires the test has
#: deadlocked and the timeout only decides how long it takes to say so.
GUARD_S = 60.0

BASE_PAYLOAD = {
    "schema_version": 2,
    "config": {"tag_throat": 2},
    "frequencies_hz": [100.0, 200.0],
}


def request_file(directory: Path, frequencies: list[float], **extra) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "request.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "config": {"tag_throat": 2},
                "cancel_path": str(directory / "cancel"),
                "frequencies_hz": frequencies,
                **extra,
            }
        ),
        encoding="utf-8",
    )
    return path


def fake_key(*, threads: str = "1") -> dict:
    return registry.worker_key(
        julia_executable=sys.executable,
        solver_script=FAKE_WORKER,
        julia_project=None,
        julia_sysimage=None,
        julia_threads=threads,
        package_version="test-0.0.0",
        package_fingerprint="fp0",
        environment={},
    )


@pytest.fixture
def staging(monkeypatch, tmp_path):
    """A private root for the staged request directories, as next door."""

    root = tmp_path / "staging"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def staged(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.name.startswith("hornlab-beat-"))


@pytest.fixture
def child_worker(monkeypatch, staging):
    monkeypatch.setenv(registry.PERSISTENT_HOST_ENV_VAR, "0")
    monkeypatch.setattr("hornlab_beat_bem.worker.default_project", lambda backend: None)
    shutdown_workers()
    yield staging
    shutdown_workers()


@pytest.fixture
def registry_dir(isolated_worker_registry, monkeypatch):
    directory = Path(tempfile.mkdtemp(prefix="o-", dir=isolated_worker_registry))
    monkeypatch.setenv(registry.WORKER_DIR_ENV_VAR, str(directory))
    yield directory
    for record in find_live_hosts(directory):
        registry.terminate_pid(record.pid)


@pytest.fixture
def hosted_worker(monkeypatch, staging, registry_dir):
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


def current_worker():
    return get_worker(
        julia_executable=sys.executable,
        solver_script=FAKE_WORKER,
        julia_threads="1",
        julia_project=None,
    )


# --------------------------------------------------------------------------
# the worker's own rule: a retirement releases nothing it does not own
# --------------------------------------------------------------------------


def test_terminating_a_child_worker_leaves_a_later_submission_locked(child_worker):
    """``terminate`` may stop the runtime; it may not free somebody's slot.

    This is the whole defect in four lines. The submission belongs to a
    request that is under way; the ``terminate`` is the tail of an *earlier*
    request that has already let go. Freeing the slot here lets a third
    request in beside the second one, and a worker that serves one job at a
    time then has two.
    """

    worker = BeatWorkerProcess(
        julia_executable=sys.executable,
        solver_script=FAKE_WORKER,
        julia_threads="1",
        julia_project=None,
    )
    try:
        worker.ensure_started()
        stream = worker.submit(request_file(Path(child_worker) / "held", [100.0, 200.0]))
        assert worker._lock.locked()

        worker.terminate()

        assert worker._lock.locked(), "a foreign terminate released the submission"
        # The owner, and only the owner, gives it back.
        stream.close()
        assert not worker._lock.locked()
    finally:
        worker.terminate()


def test_terminating_a_hosted_worker_leaves_a_later_submission_locked(registry_dir, tmp_path):
    """The hosted client carries its own copy of the same rule."""

    worker = HostedBeatWorker(fake_key(), directory=registry_dir)
    try:
        worker.ensure_started()
        stream = worker.submit(request_file(tmp_path / "hosted", [100.0, 200.0]))
        assert worker._lock.locked()

        worker.terminate()

        assert worker._lock.locked(), "a foreign terminate released the submission"
        stream.close()
        assert not worker._lock.locked()
    finally:
        worker.shutdown()


# --------------------------------------------------------------------------
# the session's rule: retirement happens while the submission is still ours
# --------------------------------------------------------------------------


def _admit_a_later_request_when_the_slot_frees(monkeypatch, worker, request_path, later):
    """Put a second request in the window an abandoned session opens.

    The window is real and it is short: closing the event stream is what frees
    the submission slot, and in the unfixed code the retirement came *after*
    that. The hook does not create the race, it only decides when the second
    request arrives -- deterministically, on an event, so the test states an
    ordering rather than betting on one.
    """

    admitted = threading.Event()

    def later_request() -> None:
        try:
            later["stream"] = worker.submit(request_path)
            # The runtime this request was admitted to. A retirement that is
            # atomic can only ever hand over one that is already replaced.
            later["worker_pid"] = getattr(worker, "pid", None)
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion
            later["error"] = exc
        finally:
            admitted.set()

    thread = threading.Thread(target=later_request, daemon=True)
    original = BeatSolveSession._close_events

    def close_events(events) -> None:
        original(events)
        thread.start()
        assert admitted.wait(GUARD_S), "the later request never reached the worker"

    monkeypatch.setattr(BeatSolveSession, "_close_events", staticmethod(close_events))
    return thread


def _abandon_a_session_while_a_later_request_waits(monkeypatch, staging_root) -> dict:
    """Run the concurrent-waiter scenario once and report what B received.

    Session A is abandoned mid-solve, which is the one ordinary path that
    retires a worker. Request B takes the submission slot the moment A gives
    it up. The two claims that follow -- A released nothing of B's, and A
    retired nothing of B's -- are asserted by separate tests, because on the
    unfixed code *both* are wrong and a single test can only report the first
    one it reaches. Split, each is its own red.
    """

    worker = current_worker()
    worker.ensure_started()
    doomed = getattr(worker, "pid", None)

    session = make_session(
        payload={"frequencies_hz": [100.0, 200.0, 300.0], "fake_wait_for_cancel": True}
    )
    stream = session.events()
    assert str(next(stream).get("type")) == "result"

    later: dict = {}
    thread = _admit_a_later_request_when_the_slot_frees(
        monkeypatch, worker, request_file(Path(staging_root) / "later", [100.0, 200.0]), later
    )

    session.close()
    thread.join(GUARD_S)

    assert "error" not in later, later.get("error")
    later["worker"] = worker
    later["doomed"] = doomed
    return later


def test_an_abandoned_session_does_not_unlock_the_next_request(child_worker, monkeypatch):
    """A must not release B's submission -- the ownership half of the rule.

    A let go of the worker when its own stream closed. Everything it does
    afterwards is somebody else's business, and the release it still attempts
    on its way out has to be refused rather than applied to whoever holds the
    slot now.
    """

    later = _abandon_a_session_while_a_later_request_waits(monkeypatch, child_worker)
    worker = later["worker"]

    assert worker._lock.locked(), "the abandoned session released the next request's submission"
    # The owner, and only the owner, gives it back.
    later["stream"].close()
    assert not worker._lock.locked()
    assert staged(child_worker) == []


def test_an_abandoned_session_does_not_retire_the_next_requests_worker(
    child_worker, monkeypatch
):
    """A must not kill the runtime B is solving on -- the ordering half.

    Stated as an outcome rather than as a schedule: the retirement finished
    before the slot changed hands, so the request that took it cannot have
    been handed the runtime that was about to be destroyed. B therefore holds
    a different child process, and its solve runs to completion.
    """

    later = _abandon_a_session_while_a_later_request_waits(monkeypatch, child_worker)

    assert later["doomed"] is not None
    assert later["worker_pid"] not in (
        None,
        later["doomed"],
    ), "the later request was admitted to the runtime the earlier one condemned"
    types = [str(event.get("type")) for event in later["stream"]]
    assert types[-1] == "completed", types


def test_an_abandoned_hosted_session_does_not_unlock_the_next_request(
    hosted_worker, monkeypatch
):
    """The hosted client carries its own copy of the ownership half."""

    later = _abandon_a_session_while_a_later_request_waits(monkeypatch, hosted_worker)
    worker = later["worker"]

    assert worker._lock.locked(), "the abandoned session released the next request's submission"
    later["stream"].close()
    assert not worker._lock.locked()


def test_an_abandoned_hosted_session_does_not_retire_the_next_requests_solve(
    hosted_worker, monkeypatch
):
    """The ordering half through the detached host, which retires over a socket.

    There is no second engine pid to compare here -- the client never
    handshakes again on the submission path -- but the host makes the damage
    louder instead: the earlier session's retirement closes the connection the
    two requests share, so an unfixed run does not merely unlock B, it ends
    B's event stream in the middle of a solve the host is still running.
    """

    later = _abandon_a_session_while_a_later_request_waits(monkeypatch, hosted_worker)

    types = [str(event.get("type")) for event in later["stream"]]
    assert types[-1] == "completed", types


# --------------------------------------------------------------------------
# a retirement that fails is still a retirement that gives the slot back
# --------------------------------------------------------------------------


def test_a_close_whose_retirement_raises_still_returns_the_submission(
    child_worker, monkeypatch
):
    """The step that can fail must not be the step that owns the unwind.

    Killing a runtime is the one part of ``close`` with a real failure mode:
    a child that survives its SIGKILL for two seconds raises
    ``TimeoutExpired``. Everything after it in an unprotected sequence --
    closing the stream, releasing the submission, deleting the staged
    directory -- is then skipped, and the session has already marked itself
    closed, so nothing will run them later either. That is not a lost solve
    but a lost *worker*: the slot stays held by a request that no longer
    exists, and every later submission waits on it forever.
    """

    worker = current_worker()
    worker.ensure_started()
    session = make_session(
        payload={"frequencies_hz": [100.0, 200.0, 300.0], "fake_wait_for_cancel": True}
    )
    stream = session.events()
    assert str(next(stream).get("type")) == "result"
    assert worker._lock.locked()

    # Raise once, then behave: the fixture's shutdown has to be able to stop
    # this worker afterwards, and a kill that fails twice is not the case
    # under test.
    kill = worker._kill_process
    attempts: list[int] = []

    def refuses_to_die_once() -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise subprocess.TimeoutExpired(cmd="julia", timeout=2.0)
        kill()

    monkeypatch.setattr(worker, "_kill_process", refuses_to_die_once)

    # The failure is reported rather than swallowed -- a runtime that will not
    # stop is worth knowing about -- but it is reported *after* the unwind.
    with pytest.raises(subprocess.TimeoutExpired):
        session.close()

    assert attempts, "the retirement never reached the kill"
    assert not worker._lock.locked(), "a failed retirement kept the submission"
    assert staged(child_worker) == [], "a failed retirement leaked the staged request"

    later = worker.submit(request_file(Path(child_worker) / "later", [100.0]))
    assert worker._lock.locked()
    later.close()


# --------------------------------------------------------------------------
# the way out of a slot whose owner is never coming back
# --------------------------------------------------------------------------


def test_shutdown_wakes_a_request_parked_on_a_worker_it_stops(child_worker):
    """A parked waiter is drained by shutdown, and not by being admitted.

    The queue behind a submission has no timeout, deliberately: a queued solve
    is meant to wait for the one in front of it. The cost is that a request
    which stops making progress parks every later one indefinitely, and
    nothing but its owner may free the slot. ``shutdown_workers`` is the
    answer, and *how* it answers matters: the waiter must be told the worker
    is gone, not handed a worker that is being terminated underneath it.
    """

    from hornlab_beat_bem.submission import SubmissionClosed

    worker = current_worker()
    worker.ensure_started()
    # Taken and never given back: a solve thread that is stuck, an unwind
    # that lost its release. The waiter behind it is the one under test.
    held = worker.submit(request_file(Path(child_worker) / "held", [100.0, 200.0]))

    outcome: dict = {}

    def later_request() -> None:
        try:
            outcome["stream"] = worker.submit(
                request_file(Path(child_worker) / "later", [100.0])
            )
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion
            outcome["error"] = exc

    thread = threading.Thread(target=later_request, daemon=True)
    thread.start()
    assert worker._lock.await_waiters(1, GUARD_S), "the later request never parked"

    shutdown_workers()

    thread.join(GUARD_S)
    assert not thread.is_alive(), "shutdown left a request parked on a stopped worker"
    assert "stream" not in outcome, "a worker being stopped admitted a submission"
    assert isinstance(outcome.get("error"), SubmissionClosed), outcome.get("error")
    held.close()


# --------------------------------------------------------------------------
# one host, two clients: whose runtime is it
# --------------------------------------------------------------------------


def test_one_clients_retire_does_not_end_another_clients_solve(
    registry_dir, tmp_path, monkeypatch
):
    """The ownership rule, at the one boundary the slot cannot reach.

    Two applications adopting one host is what the host exists for, and their
    submissions are ordered by the host rather than by either client's slot.
    A ``retire`` frame therefore arrives with no proof of ownership attached
    -- and it always arrives on a connection whose own job has ended, because
    a connection is either inside a solve or reading frames. So a job running
    when one lands belongs to somebody else, and the host declines rather than
    killing a runtime it has already handed on in good order.
    """

    # The stand-in worker's cancel wait is a deadlock guard here, not a race:
    # the cancel below is written only after a second client has been served.
    monkeypatch.setenv("FAKE_BEAT_CANCEL_WAIT_S", str(GUARD_S))

    solving = HostedBeatWorker(fake_key(), directory=registry_dir)
    other = HostedBeatWorker(fake_key(), directory=registry_dir)
    try:
        solving.ensure_started()
        engine = solving.engine_pid
        other.ensure_started()
        assert other.engine_pid == engine, "the two clients adopted different hosts"

        directory = tmp_path / "solving"
        stream = solving.submit(
            request_file(directory, [100.0, 200.0], fake_wait_for_cancel=True)
        )
        # Reading a result is what establishes that the host is inside this
        # job: the engine is now waiting for the cancel file, and the host's
        # job lock is held for the whole of it.
        types = []
        for event in stream:
            types.append(str(event.get("type")))
            if types[-1] == "result":
                break
        assert types[-1] == "result", types

        # Through the public call, not a hand-rolled frame: what a client
        # asks for is a retirement, and what it gets back is whether it
        # happened. Declined here, because the other client is mid-solve.
        assert other.terminate() is False

        (directory / "cancel").write_text("cancel", encoding="utf-8")
        types.extend(str(event.get("type")) for event in stream)
        assert types[-1] == "cancelled", types

        solving.ensure_started()
        assert solving.engine_pid == engine, "the other client's retire killed the runtime"

        # And the same call, with nobody else solving, does retire it: the
        # answer is the host's, not a constant False that would make the
        # assertion above pass for the wrong reason.
        other.ensure_started()
        assert other.terminate() is True
    finally:
        other.detach()
        solving.shutdown()


# --------------------------------------------------------------------------
# the primitive
# --------------------------------------------------------------------------


def test_a_submission_slot_refuses_a_release_from_anyone_but_the_holder():
    from hornlab_beat_bem.submission import SubmissionSlot

    slot = SubmissionSlot()
    mine = slot.acquire()

    assert slot.holds(mine)
    assert not slot.release(object()), "a foreign object must not free the slot"
    assert slot.locked()
    assert slot.release(mine)
    assert not slot.locked()
    # Releasing twice is what an unwind that runs after a generator's finally
    # does, and it must not free whatever came next.
    later = slot.acquire()
    assert not slot.release(mine)
    assert slot.holds(later)
    assert slot.release(later)


def test_a_submission_slot_hands_over_only_after_a_retirement_finishes():
    """Atomic retirement: no waiter is admitted while the runtime is dying.

    The action stands in for killing the Julia child. A waiter admitted while
    it runs would be handed exactly the runtime being destroyed, which is the
    ordering half of this defect.
    """

    from hornlab_beat_bem.submission import SubmissionSlot

    slot = SubmissionSlot()
    mine = slot.acquire()

    inside = threading.Event()
    finish = threading.Event()
    admitted = threading.Event()
    retired: dict = {}

    def action() -> None:
        inside.set()
        assert finish.wait(GUARD_S)

    def waiter() -> None:
        ticket = slot.acquire()
        admitted.set()
        slot.release(ticket)

    threading.Thread(target=waiter, daemon=True).start()
    # Without this the assertion below is vacuous: a waiter that had not yet
    # reached ``acquire`` is not admitted either, and the test would pass on
    # code that hands the slot over mid-retirement.
    assert slot.await_waiters(1, GUARD_S), "the waiter never parked on the slot"
    retirement = threading.Thread(
        target=lambda: retired.setdefault("ran", slot.retire(mine, action)), daemon=True
    )
    retirement.start()

    assert inside.wait(GUARD_S)
    assert not admitted.is_set(), "the slot changed hands during a retirement"
    finish.set()
    retirement.join(GUARD_S)
    assert retired["ran"] is True
    assert not admitted.is_set(), "the retirement's owner still holds the slot"

    slot.release(mine)
    assert admitted.wait(GUARD_S)


def test_a_closed_slot_wakes_its_waiters_and_admits_nobody_after():
    """Closing ends the queue without confiscating the submission.

    Both halves are the point. A waiter has to leave -- by raising, which is
    the only honest answer to "give me a worker that has been stopped" -- and
    the current holder has to keep what it holds, or closing would be the very
    hand-over the slot exists to prevent.
    """

    from hornlab_beat_bem.submission import SubmissionClosed, SubmissionSlot

    slot = SubmissionSlot()
    mine = slot.acquire()

    outcome: dict = {}

    def waiter() -> None:
        try:
            outcome["ticket"] = slot.acquire()
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion
            outcome["error"] = exc

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    assert slot.await_waiters(1, GUARD_S), "the waiter never parked on the slot"

    slot.close()

    thread.join(GUARD_S)
    assert not thread.is_alive(), "closing the slot left a waiter parked"
    assert "ticket" not in outcome, "a closed slot admitted a submission"
    assert isinstance(outcome.get("error"), SubmissionClosed), outcome.get("error")

    assert slot.closed()
    assert slot.holds(mine), "closing took the slot away from its holder"
    assert slot.release(mine)
    with pytest.raises(SubmissionClosed):
        slot.acquire()


def test_a_stale_owner_cannot_retire_the_slot_it_no_longer_holds():
    from hornlab_beat_bem.submission import SubmissionSlot

    slot = SubmissionSlot()
    mine = slot.acquire()
    assert slot.release(mine)
    later = slot.acquire()

    ran: list[str] = []
    assert not slot.retire(mine, lambda: ran.append("killed"))
    assert ran == [], "an earlier request retired a later one's worker"
    assert slot.holds(later)
    assert slot.release(later)


def test_a_dropped_streams_finalizer_gives_up_rather_than_wait_for_the_mutex():
    """The collector chooses the thread; it may choose one inside the mutex.

    An abandoned hosted stream is cycle garbage -- session, events, generator
    frame and bound status callback close a loop -- so its finalizer runs on
    whatever thread the cyclic collector interrupts, at whatever allocation it
    interrupts it. ``retire`` holds the slot's mutex for its whole action, and
    for the hosted worker that action is seconds of socket I/O with plenty of
    allocation in it. A finalizer landing there would ask a non-reentrant
    mutex for a lock its own thread is holding.

    The collection is spelled out here rather than provoked, because a test
    that waits for a real ``gc.collect()`` to land on the right instruction is
    a test that passes for timing reasons. Calling the finalizer from inside
    the retirement is that instruction, deterministically.
    """

    from hornlab_beat_bem.submission import Submission, SubmissionSlot, SubmissionStream

    slot = SubmissionSlot()
    #: A stream left over from a request that no longer owns anything: what a
    #: dropped stream is by the time somebody else is retiring the slot.
    stale = Submission(0)
    released_in_the_generator: list[bool] = []

    def events():
        try:
            yield {}
        finally:
            # Closing the stream runs this, and it releases the same slot --
            # so the guard has to cover the whole of ``close``, not just the
            # release the stream makes itself.
            released_in_the_generator.append(slot.release(stale))

    abandoned = SubmissionStream(slot, stale, events())
    next(abandoned)  # start it, so that ``finally`` is live

    mine = slot.acquire()
    finished = threading.Event()

    def retire() -> None:
        slot.retire(mine, abandoned.__del__)
        finished.set()

    thread = threading.Thread(target=retire, daemon=True)
    thread.start()
    assert finished.wait(GUARD_S), "the finalizer waited for a mutex its own thread held"
    thread.join(GUARD_S)

    assert released_in_the_generator == [False], released_in_the_generator
    assert slot.holds(mine), "a finalizer freed the retiring request's slot"

    # Giving up is only for the contended case. Uncontended -- the window the
    # finalizer exists for, an exception between ``submit`` returning and the
    # caller storing what it returned -- it still gives the slot back.
    assert slot.release(mine)
    ours = slot.acquire()
    SubmissionStream(slot, ours, events()).__del__()
    assert not slot.locked(), "an uncontended finalizer kept the slot"
