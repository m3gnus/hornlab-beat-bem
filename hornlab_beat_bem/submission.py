"""Ownership of a worker's single submission slot.

A BEAT worker serves one job at a time, and both worker implementations used
to enforce that with a bare ``threading.Lock``. A bare lock records that it is
held; it does not record *by whom*, and every retirement path in this package
needs exactly that missing fact. ``terminate`` asked ``locked()`` and released
whatever it found, so an earlier request that had already let go could free the
next request's submission -- and once two callers believe they hold a worker
that serves one job, the second one's events are the first one's.

The slot below answers the question the lock could not:

``acquire`` issues a :class:`Submission`
    an identity, not a boolean. Holding it is what makes the worker yours.
``release`` is refused to anyone but the holder
    so a late unwind -- a generator finalized after the fact, a second
    ``close`` -- is a no-op instead of a hand-over.
``retire`` runs its action only while the caller still holds the slot, and
holds it throughout
    which is the ordering half of the same defect: releasing the slot and
    *then* killing the runtime leaves a window in which the next request is
    admitted to a worker that is already condemned. Here there is no window,
    because the slot never becomes free during the retirement.
``close`` ends the slot for everyone
    the counterweight to the rule above. Ownership means nobody but the owner
    can free the slot, so a holder that never gives it back parks every later
    request forever -- and ``acquire`` has no timeout, because a queued solve
    is meant to wait for the one in front of it. ``close`` is the terminal
    state that answers that: it admits nobody, wakes everybody, and makes
    every ``acquire`` from then on raise :class:`SubmissionClosed` instead of
    waiting for a worker this process has let go of. It deliberately does
    *not* take the slot away from its current holder -- that would be the
    hand-over this module exists to prevent.

Nothing in here knows what a worker is. It is deliberately a small,
directly testable primitive: the two worker classes and the host all make the
same ownership mistakes, and they should not each carry their own repair.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


class SubmissionClosed(RuntimeError):
    """The slot has been closed; it will never admit another submission.

    Raised rather than returned because there is no useful degraded answer: a
    caller asking for a worker that this process has stopped cannot proceed,
    and the alternative -- waiting -- is the hang this exception replaces.
    """


class Submission:
    """The right to drive one worker, issued to exactly one caller.

    Identity is the whole content: two submissions are never equal and a
    submission cannot be reconstructed, so possession of the object is the
    proof of ownership. The serial exists for logs and test failures.
    """

    __slots__ = ("serial",)

    def __init__(self, serial: int) -> None:
        self.serial = serial

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<Submission {self.serial}>"


class SubmissionSlot:
    """One submission at a time, with the holder recorded.

    The condition's mutex is held for the whole of ``retire``, which is what
    makes a retirement atomic against a waiting ``acquire``: a waiter is
    parked in ``wait()`` (which releases the mutex) or blocked on the mutex
    itself, and in neither case can it be admitted while the action runs. The
    action must therefore not call back into this slot -- and neither may a
    finalizer that the collector happens to run on the same thread, which is
    what :meth:`best_effort_release` is for.
    """

    def __init__(self) -> None:
        mutex = threading.Lock()
        self._changed = threading.Condition(mutex)
        # A second waitset on the *same* mutex, so the two never see
        # inconsistent state. It exists so that announcing "a waiter has
        # parked" cannot wake the waiters themselves: notifying them on
        # ``_changed`` would have every waiter re-park and re-announce, and
        # two waiters would spin against each other forever.
        self._parked = threading.Condition(mutex)
        self._holder: Submission | None = None
        self._serials = itertools.count(1)
        self._closed = False
        self._waiting = 0
        # Set on the thread inside ``best_effort_release`` -- see there.
        self._best_effort = threading.local()

    def acquire(self) -> Submission:
        """Wait for the slot and take it. Returns the proof of ownership.

        Raises :class:`SubmissionClosed` if the slot is closed, or is closed
        while this caller is waiting.
        """

        with self._changed:
            while True:
                if self._closed:
                    raise SubmissionClosed(
                        "This worker has been stopped and admits no further submissions."
                    )
                if self._holder is None:
                    break
                self._waiting += 1
                # Announce the parking as well as wait for the slot:
                # ``await_waiters`` below is how a caller -- a test, mostly --
                # establishes that somebody really is queued before asserting
                # what happens to a queue.
                self._parked.notify_all()
                try:
                    self._changed.wait()
                finally:
                    self._waiting -= 1
            self._holder = Submission(next(self._serials))
            return self._holder

    def close(self) -> None:
        """Admit nothing more, and wake everyone waiting to be told so.

        The terminal state ``acquire`` otherwise lacks. It is what a process
        letting go of a worker calls, and it is idempotent. The current
        holder keeps the slot and is still free to ``release`` or ``retire``
        it: closing the slot ends the *queue*, it does not confiscate a
        submission that somebody is still using.
        """

        with self._changed:
            self._closed = True
            self._changed.notify_all()

    def closed(self) -> bool:
        """Whether the slot has been closed. Diagnostics and tests only."""

        with self._changed:
            return self._closed

    def await_waiters(self, count: int = 1, timeout: float | None = None) -> bool:
        """Block until at least ``count`` callers are parked in ``acquire``.

        Observability for the concurrency this class exists to order. A test
        asserting "nobody was admitted while X ran" is worth nothing unless a
        waiter had actually arrived, and the alternative ways to establish
        that are a sleep or a poll -- one flaky, the other slow.

        A waiter increments the count under the mutex and only lets go of it
        inside ``wait``, so seeing the count from here means the waiter is
        genuinely parked rather than about to be.
        """

        with self._parked:
            return self._parked.wait_for(lambda: self._waiting >= count, timeout)

    def release(self, submission: object) -> bool:
        """Give the slot back. A stale or foreign submission changes nothing.

        Returning ``False`` rather than raising is the point: the callers are
        unwind paths that run in an order nobody controls, and "somebody else
        owns this now" is a normal outcome there, not an error.

        The parameter is typed ``object`` rather than ``Submission | None``
        for the same reason: the check is identity against the current holder,
        so *anything* that is not it is refused, and a caller holding a stale
        or foreign object -- which is what the ownership tests hand it -- is
        exactly the case being defended against rather than a type error.

        Inside :meth:`best_effort_release` this gives up rather than waits for
        the mutex, and returns ``False`` for that too.
        """

        if not self._changed.acquire(blocking=self._may_block()):
            # Best-effort mode, and somebody -- possibly this very thread --
            # is inside the mutex. "Not mine to free" and "not free-able just
            # now" are the same answer to the same callers.
            return False
        try:
            if submission is None or self._holder is not submission:
                return False
            self._holder = None
            self._changed.notify()
            return True
        finally:
            self._changed.release()

    def _may_block(self) -> bool:
        return not getattr(self._best_effort, "on", False)

    @contextmanager
    def best_effort_release(self) -> Iterator[None]:
        """Within this block, on this thread, no release waits for the mutex.

        For finalizers, and for nothing else. A finalizer runs on whichever
        thread the cyclic collector happened to interrupt, at whichever
        allocation it happened to interrupt it -- including a thread that is
        *inside* this slot's mutex, which :meth:`retire` holds for the whole
        of a retirement (seconds of socket I/O, for the hosted worker). The
        mutex is not reentrant, so a release from there would wait for a lock
        its own thread holds and never get it.

        Making the mutex reentrant instead would trade the hang for something
        worse: a finalizer would then be able to clear ``_holder`` in the
        middle of a retirement, which is the hand-over this module exists to
        prevent. Giving up is the safe half of that choice. What it costs is
        a slot that stays held by a stream nobody is reading until somebody
        closes it explicitly -- and ``close`` on the slot is exactly the way
        out of that, already taken by ``shutdown_workers``.

        The flag is per-thread, and it covers *nested* releases as well as
        the one the caller can see: closing a stream runs its generator's
        ``finally``, which releases this same slot from inside the finalizer
        that must not block.
        """

        previous = getattr(self._best_effort, "on", False)
        self._best_effort.on = True
        try:
            yield
        finally:
            self._best_effort.on = previous

    def retire(
        self, submission: Submission | None, action: Callable[[], Any]
    ) -> bool:
        """Run ``action`` if -- and while -- ``submission`` still owns the slot.

        The slot is *not* released afterwards. The owner releases it when it
        is done, which is what keeps the retirement and the hand-over from
        being two separately observable events.
        """

        with self._changed:
            if submission is None or self._holder is not submission:
                return False
            action()
            return True

    def holds(self, submission: Submission | None) -> bool:
        with self._changed:
            return submission is not None and self._holder is submission

    def locked(self) -> bool:
        """Whether anyone holds the slot. Diagnostics and tests only.

        Deliberately not a basis for a decision: "somebody holds it" is the
        very question that produced the defect this module exists to fix.
        """

        with self._changed:
            return self._holder is not None

    @contextmanager
    def held(self) -> Iterator[Submission]:
        """Own the slot for the duration of a block that is not a submission."""

        submission = self.acquire()
        try:
            yield submission
        finally:
            self.release(submission)


class SubmissionStream:
    """A submission's events, together with the submission itself.

    ``submit`` has to hand back two things -- the events, and the right to end
    them -- and it used to hand back only the events. The caller was left
    saying "retire the worker" when the only safe sentence is "retire *my*
    submission", which is the one this type lets it say.

    It is an iterator and nothing more: callers iterate it, ``close`` it, and
    (for the host) hand it to a loop that does both, exactly as they did with
    the generator it wraps.

    ``close`` releases the submission as well as closing the events, because
    the two are not the same thing. A generator that was never started -- a
    caller that failed between ``submit`` and its first event -- closes
    without ever entering its ``finally``, and the slot would stay held by a
    request that no longer exists.
    """

    __slots__ = ("submission", "_slot", "_events")

    def __init__(
        self, slot: SubmissionSlot, submission: Submission, events: Iterator[dict]
    ) -> None:
        self.submission = submission
        self._slot = slot
        self._events = events

    def __iter__(self) -> SubmissionStream:
        return self

    def __next__(self) -> dict:
        return next(self._events)

    def close(self) -> None:
        try:
            close = getattr(self._events, "close", None)
            if close is not None:
                close()
        finally:
            self._slot.release(self.submission)

    def __del__(self) -> None:
        """Last resort for a stream that was created and then dropped.

        ``submit`` returns the slot and the events in one object precisely so
        that the caller cannot take one without the other -- but there is
        still an instant between the return and the caller storing it, and an
        asynchronous exception (an interrupt, a timeout signal) landing in
        that instant would drop a stream nobody can close, holding the slot
        until nothing but a garbage collection could free it. Here that
        collection is the release.

        **This may not block, and so it may skip.** An abandoned hosted stream
        is cycle garbage rather than a refcount drop -- session, events,
        generator frame and bound status callback form a loop -- so the
        collector runs this on an arbitrary thread at an arbitrary allocation,
        including one inside the slot's mutex during a retirement. Every
        release this path performs therefore runs under
        :meth:`SubmissionSlot.best_effort_release`: it gives up rather than
        waits, here and in the generator ``finally`` that ``close`` runs. What
        that leaves behind when it does give up is a slot still held by a
        stream nobody is reading, freed by an explicit ``close`` on the stream
        or, at shutdown, on the slot. That is the cheaper failure: an
        occasional slot held until then costs the next solve a wait it can be
        released from, where blocking here would hang the retiring thread with
        no way out at all.

        Bare ``except`` rather than ``contextlib.suppress`` because this can
        run during interpreter shutdown, when the module's own globals may
        already be gone.
        """

        try:
            with self._slot.best_effort_release():
                self.close()
        except BaseException:  # noqa: BLE001 - a finalizer may not raise
            pass
