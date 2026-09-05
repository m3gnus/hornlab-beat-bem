"""Readiness is per backend, and provisioning one never un-provisions another.

The defect these tests pin was reported from a Fedora host with an RTX 5090:
``beat-cuda`` was available, ``beat-cpu`` was permanently unavailable, and the
command the interface offered to fix that -- ``provision --backend cpu`` --
wrote the single ``state.json`` the CUDA record lived in. One file with one
``backend`` field could not represent the both-provisioned outcome at all, so a
selector that offers the four backends as four engines had one row that could
never light up on a GPU machine.

Two properties are asserted here, against each other:

* **Independence.** Each backend's record answers only for that backend, in
  both directions, and a version-1 ``state.json`` is still honoured for the
  backend it names so an already-provisioned host does not re-download Julia.
* **Serialisation.** Provisioning is not per backend in one respect -- it
  unpacks a shared portable Julia, replacing an existing unpack -- so two runs
  hold a lock rather than interleave, and a holder that dies does not block the
  machine forever.

Nothing here reaches the network or starts Julia: ``_run_julia_step`` and
``_download`` are replaced, exactly as ``test_provision.py`` does it.
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hornlab_beat_bem import provision, runtime
from hornlab_beat_bem.config import BEAT_CPU, BEAT_CUDA, BEAT_METAL, BEAT_ROCM
from hornlab_beat_bem.runtime import PACKAGE_DIR


@pytest.fixture()
def runtime_dir(tmp_path, monkeypatch):
    directory = tmp_path / "runtime"
    monkeypatch.setenv(provision.RUNTIME_DIR_ENV_VAR, str(directory))
    monkeypatch.delenv(runtime.FORCE_CPU_ENV_VAR, raising=False)
    monkeypatch.delenv(runtime.JULIA_ENV_VAR, raising=False)
    return directory


@pytest.fixture()
def fake_julia(monkeypatch, tmp_path):
    executable = tmp_path / "julia" / "bin" / "julia"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_julia", lambda explicit=None: str(executable))

    def forbidden(*args, **kwargs):
        raise AssertionError("a download was attempted although Julia exists")

    monkeypatch.setattr(provision, "_download", forbidden)
    return executable


@pytest.fixture()
def julia_steps(monkeypatch):
    calls: list[dict] = []

    def record(julia, code, **kwargs):
        calls.append({"julia": julia, "code": code, **kwargs})

    monkeypatch.setattr(provision, "_run_julia_step", record)
    return calls


@pytest.fixture()
def cuda_host(monkeypatch):
    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setattr(runtime, "_rocm_present", lambda: False)
    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: False)


@pytest.fixture()
def no_gpu(monkeypatch):
    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setattr(runtime, "_rocm_present", lambda: False)
    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: False)


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition was not reached in time")


#: A real second process holding the lock, because the property under test is
#: the operating system's: the lock is released when its holder dies, whatever
#: killed it and whatever the holder had recorded about itself.
_LOCK_HOLDER_SOURCE = """
import sys, time
from pathlib import Path
from hornlab_beat_bem import provision

with provision._provisioning_lock(
    Path(sys.argv[1]), backend="cuda", status_cb=lambda message: None
):
    print("locked", flush=True)
    time.sleep(120)
"""


def _lock_holding_subprocess(runtime_dir) -> subprocess.Popen:
    repo_root = PACKAGE_DIR.parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    process = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER_SOURCE, str(runtime_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        cwd=str(repo_root),
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    if line.strip() != "locked":  # pragma: no cover - a broken child is a test bug
        process.kill()
        raise AssertionError(
            f"the lock-holding subprocess did not start: {line!r} "
            f"{process.stderr.read() if process.stderr else ''}"
        )
    return process


def _legacy_state(runtime_dir, backend: str, julia, **extra) -> None:
    """A version-1 record: one file, one backend, no ``state_schema`` key."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    project = runtime.default_project(backend)
    payload = {
        "status": "ready",
        "backend": backend,
        "project": str(project),
        "package_fingerprint": runtime.package_fingerprint(project),
        "julia_executable": str(julia),
        "step": "done",
    }
    payload.update(extra)
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Independence: one record per backend.
# ---------------------------------------------------------------------------


def test_provisioning_the_cpu_leaves_a_provisioned_gpu_ready(
    runtime_dir, cuda_host, fake_julia, julia_steps
):
    """The reported defect, as an assertion.

    A user on a CUDA box follows the "provision the CPU runtime" command in a
    greyed-out row's reason. Afterwards *both* backends must be ready: the
    whole point of offering the CPU path as its own engine is that it is not
    bought with the GPU one.
    """

    assert provision.provision_gpu(
        runtime_dir, backend=BEAT_CUDA, status_cb=lambda _: None
    )["status"] == "ready"
    assert provision.provision_cpu(runtime_dir, status_cb=lambda _: None)["status"] == "ready"

    assert provision.backend_ready(BEAT_CUDA) is True
    assert provision.backend_ready(BEAT_CPU) is True
    states = provision.read_backend_states(runtime_dir)
    assert {name: state["status"] for name, state in states.items()} == {
        BEAT_CUDA: "ready",
        BEAT_CPU: "ready",
    }


def test_a_cpu_provisioning_does_not_make_the_gpu_re_instantiate(
    runtime_dir, cuda_host, fake_julia, julia_steps
):
    """Nor is the GPU quietly re-provisioned to recover from the CPU write.

    Before per-backend records the CPU run overwrote the CUDA record, so the
    next GPU hook re-ran a multi-gigabyte artifact resolution to get back to
    where it already was. That cost is invisible in a capability row.
    """

    provision.provision_gpu(runtime_dir, backend=BEAT_CUDA, status_cb=lambda _: None)
    provision.provision_cpu(runtime_dir, status_cb=lambda _: None)
    julia_steps.clear()

    assert provision.provision_gpu(
        runtime_dir, backend=BEAT_CUDA, status_cb=lambda _: None
    )["status"] == "ready"
    assert julia_steps == [], "the CUDA runtime was re-instantiated"


def test_each_backend_reads_only_its_own_record(runtime_dir, no_gpu, fake_julia, julia_steps):
    provision.provision_cpu(runtime_dir, status_cb=lambda _: None)

    assert provision.read_state(runtime_dir, backend=BEAT_CPU)["status"] == "ready"
    for backend in (BEAT_CUDA, BEAT_ROCM, BEAT_METAL):
        assert provision.read_state(runtime_dir, backend=backend) is None
        assert provision.backend_ready(backend) is False


def test_a_version_1_record_is_still_honoured_for_the_backend_it_names(
    runtime_dir, cuda_host, fake_julia, julia_steps
):
    """A host provisioned before the split must not re-download anything.

    The legacy file is the only record such a machine has, and it names one
    backend. Read as that backend's record it is exactly right; read as an
    answer about any other backend it is exactly wrong.
    """

    _legacy_state(runtime_dir, BEAT_CUDA, fake_julia)

    assert provision.backend_ready(BEAT_CUDA) is True
    assert provision.backend_ready(BEAT_CPU) is False
    assert provision.provision_gpu(
        runtime_dir, backend=BEAT_CUDA, status_cb=lambda _: None
    )["status"] == "ready"
    assert julia_steps == [], "a legacy ready record was ignored and re-provisioned"


def test_the_legacy_mirror_keeps_answering_an_older_consumer(
    runtime_dir, no_gpu, fake_julia, julia_steps
):
    """``read_state()`` with no backend is a pinned consumer's call shape.

    Waveguide Generator at the currently pinned commit reads the single file and
    then matches ``backend``/``project``/``package_fingerprint`` itself. The
    mirror is what keeps that consumer working without a pin bump, and its own
    matching rule is what keeps the mirror from over-promising: a mirror
    describing another backend fails the consumer's test.
    """

    provision.provision_cpu(runtime_dir, status_cb=lambda _: None)

    mirrored = provision.read_state(runtime_dir)
    assert mirrored is not None
    assert mirrored["backend"] == BEAT_CPU
    assert mirrored["status"] == "ready"
    assert mirrored["project"] == str(PACKAGE_DIR / "julia")
    assert mirrored["package_fingerprint"] == runtime.package_fingerprint(
        PACKAGE_DIR / "julia"
    )
    assert mirrored["state_schema"] == provision.STATE_SCHEMA_VERSION


def test_a_julia_is_found_through_any_ready_record(
    runtime_dir, cuda_host, fake_julia, julia_steps, monkeypatch
):
    """Discovery's third tier must not be lost to another backend's failure.

    The CPU runtime is ready and a later CUDA attempt failed, so the mirror
    describes a failure. A Julia is not per backend -- the same executable runs
    both projects -- so the recorded one is still the right answer.
    """

    provision.provision_cpu(runtime_dir, status_cb=lambda _: None)

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA artifacts could not be resolved")

    monkeypatch.setattr(provision, "_run_julia_step", boom)
    assert provision.provision_gpu(
        runtime_dir, backend=BEAT_CUDA, status_cb=lambda _: None
    )["status"] == "failed"

    assert provision.read_state(runtime_dir)["status"] == "failed"
    assert provision.provisioned_julia(runtime_dir) == str(fake_julia)
    assert provision.provisioned_julia(runtime_dir, backend=BEAT_CUDA) is None
    assert provision.provisioned_julia(runtime_dir, backend=BEAT_CPU) == str(fake_julia)


# ---------------------------------------------------------------------------
# Serialisation: one provisioning at a time per runtime directory.
# ---------------------------------------------------------------------------


def test_a_second_provisioning_waits_instead_of_interleaving(
    runtime_dir, no_gpu, fake_julia, julia_steps
):
    """Two runs must not unpack a shared Julia at the same time.

    ``_extract_julia`` removes an existing unpack before replacing it, so an
    interleaved pair can delete the Julia the other one is executing. The
    waiting run says what it is waiting for, because a consumer surfaces these
    messages as the engine row's reason.
    """

    messages: list[str] = []
    released = threading.Event()
    holder_ready = threading.Event()

    def hold():
        with provision._provisioning_lock(
            runtime_dir, backend=BEAT_CUDA, status_cb=lambda _: None
        ):
            holder_ready.set()
            released.wait(10.0)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert holder_ready.wait(5.0)

    result: list[dict] = []
    waiter = threading.Thread(
        target=lambda: result.append(
            provision.provision_cpu(runtime_dir, status_cb=messages.append)
        ),
        daemon=True,
    )
    waiter.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any("Waiting for" in m for m in messages):
            time.sleep(0.05)
        assert any("Waiting for the BEAT cuda runtime provisioning" in m for m in messages), messages
        assert julia_steps == [], "the waiter started work while the lock was held"
        assert provision.read_state(runtime_dir, backend=BEAT_CPU) is None
    finally:
        released.set()
    waiter.join(timeout=10.0)
    holder.join(timeout=10.0)

    assert result and result[0]["status"] == "ready"
    # The lock file outlives every run that uses it, by design.
    assert (runtime_dir / provision.LOCK_FILENAME).exists()


def test_the_waiter_short_circuits_when_the_holder_provisioned_it(
    runtime_dir, no_gpu, fake_julia, julia_steps
):
    """The thing being waited for may be this very backend.

    Re-reading the record under the lock is what turns a queued duplicate into
    a no-op instead of a second instantiate.
    """

    released = threading.Event()
    holder_ready = threading.Event()
    project = runtime.default_project(BEAT_CPU)

    def hold():
        with provision._provisioning_lock(
            runtime_dir, backend=BEAT_CPU, status_cb=lambda _: None
        ):
            holder_ready.set()
            released.wait(10.0)
            provision._write_state(
                runtime_dir,
                {
                    "status": "ready",
                    "backend": BEAT_CPU,
                    "project": str(project),
                    "package_fingerprint": runtime.package_fingerprint(project),
                    "julia_executable": str(fake_julia),
                    "step": "done",
                },
            )

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert holder_ready.wait(5.0)

    result: list[dict] = []
    waiter = threading.Thread(
        target=lambda: result.append(
            provision.provision_cpu(runtime_dir, status_cb=lambda _: None)
        ),
        daemon=True,
    )
    waiter.start()
    time.sleep(0.2)
    released.set()
    waiter.join(timeout=10.0)
    holder.join(timeout=10.0)

    assert result and result[0]["status"] == "ready"
    assert julia_steps == [], "the waiter re-instantiated a runtime already made ready"


def test_a_dead_holder_releases_the_lock_without_anyone_taking_it_over(
    runtime_dir, no_gpu, fake_julia, julia_steps
):
    """A crashed provisioning must not block the machine, and does not.

    This is the property that replaced a heartbeat lease: the lock is the
    kernel's, held by an open file, so killing the holder releases it. Nothing
    estimates staleness, nothing takes anything over, and a live holder
    suspended for a minute inside a native call is therefore in no danger.
    """

    holder = _lock_holding_subprocess(runtime_dir)
    try:
        messages: list[str] = []
        result: list[dict] = []
        waiter = threading.Thread(
            target=lambda: result.append(
                provision.provision_cpu(runtime_dir, status_cb=messages.append)
            ),
            daemon=True,
        )
        waiter.start()
        _wait_until(lambda: any("Waiting for" in message for message in messages))
        assert julia_steps == [], "the waiter worked while another process held the lock"

        holder.kill()
        holder.wait(timeout=10.0)
        waiter.join(timeout=10.0)
    finally:
        if holder.poll() is None:  # pragma: no cover - only on an assertion failure
            holder.kill()

    assert result and result[0]["status"] == "ready"
    assert not any("stopped responding" in message for message in messages), messages
    # And the file it locked is still there: unlinking a lock under a waiter is
    # how two processes end up holding two different inodes.
    assert (runtime_dir / provision.LOCK_FILENAME).exists()


def test_two_threads_in_one_process_serialise_on_the_same_lock(runtime_dir):
    """``flock`` and ``msvcrt.locking`` are per open file, not per process.

    Worth an assertion rather than a citation: POSIX ``fcntl`` record locks --
    the other obvious choice -- are per *process* and would let two threads in
    one server both provision, which is exactly the shape a background
    preparation thread plus an operator's CLI can take.
    """

    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    def hold():
        with provision._provisioning_lock(
            runtime_dir, backend=BEAT_CUDA, status_cb=lambda _: None
        ):
            entered.set()
            release.wait(10.0)

    def contend():
        with provision._provisioning_lock(
            runtime_dir, backend=BEAT_CPU, status_cb=lambda _: None
        ):
            second_entered.set()

    first = threading.Thread(target=hold, daemon=True)
    first.start()
    assert entered.wait(5.0)
    second = threading.Thread(target=contend, daemon=True)
    second.start()
    try:
        assert not second_entered.wait(1.0), "two threads held the provisioning lock"
    finally:
        release.set()
    assert second_entered.wait(5.0)
    first.join(timeout=5.0)
    second.join(timeout=5.0)


def test_no_liveness_signal_is_ever_sent_to_a_recorded_pid(runtime_dir, monkeypatch):
    """``os.kill(pid, 0)`` is not a liveness query on Windows.

    There it terminates the target for every signal other than CTRL_C and
    CTRL_BREAK, so a lock implementation that probed a holder that way would
    kill whatever process happened to hold that pid. Both halves are checked:
    the module does not spell it, and a run whose ``os.kill`` explodes still
    completes -- an absence of grep hits is an absence of a spelling, never an
    absence of the behaviour.
    """

    source = Path(provision.__file__).read_text(encoding="utf-8")
    # The call, not the word: the module names it in a comment explaining why it
    # is not used, and a check that could not tell those apart would have to be
    # deleted the first time someone wrote that comment down.
    assert "os.kill(" not in source

    def forbidden(*args, **kwargs):
        raise AssertionError("a signal was sent to probe a lock holder")

    monkeypatch.setattr(os, "kill", forbidden)
    with provision._provisioning_lock(
        runtime_dir, backend=BEAT_CPU, status_cb=lambda _: None
    ):
        pass


def test_the_lock_is_released_when_provisioning_fails(
    runtime_dir, no_gpu, fake_julia, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("instantiate exploded")

    monkeypatch.setattr(provision, "_run_julia_step", boom)

    assert provision.provision_cpu(runtime_dir, status_cb=lambda _: None)["status"] == "failed"
    # Released, not removed: a second run acquires it immediately.
    assert (runtime_dir / provision.LOCK_FILENAME).exists()
    with provision._provisioning_lock(
        runtime_dir, backend=BEAT_CPU, status_cb=lambda _: None
    ):
        pass


# ---------------------------------------------------------------------------
# Migrating a host that was provisioned before the split.
# ---------------------------------------------------------------------------


def test_a_failed_cpu_run_does_not_lose_a_legacy_gpu_record(
    runtime_dir, cuda_host, fake_julia, monkeypatch
):
    """The migration order that could destroy the only record a host has.

    Legacy layout, CUDA ready, nothing else. The CPU run's first write is
    ``in_progress`` and its mirror lands on that same file -- so if the record
    were not migrated first, an offline CPU attempt would take the CUDA
    readiness and the Julia it names with it.
    """

    _legacy_state(runtime_dir, BEAT_CUDA, fake_julia)
    monkeypatch.setattr(
        runtime, "_julia_gpu_functional", lambda julia, backend: (True, "probe says yes")
    )

    def boom(*args, **kwargs):
        raise RuntimeError("urllib could not resolve julialang-s3.julialang.org")

    monkeypatch.setattr(provision, "_run_julia_step", boom)

    assert provision.provision_cpu(runtime_dir, status_cb=lambda _: None)["status"] == "failed"

    assert provision.backend_ready(BEAT_CUDA) is True
    assert provision.read_state(runtime_dir, backend=BEAT_CUDA)["julia_executable"] == str(
        fake_julia
    )
    assert provision.provisioned_julia(runtime_dir) == str(fake_julia)
    assert runtime.backend_status(BEAT_CUDA)["available"] is True
    assert runtime.backend_status(BEAT_CPU)["state"] == "failed"


def test_an_interrupted_cpu_run_does_not_lose_a_legacy_gpu_record(
    runtime_dir, cuda_host, fake_julia, monkeypatch
):
    """The same order, interrupted rather than failed.

    A machine switched off mid-provisioning leaves ``in_progress`` behind. The
    CUDA record must still be there when it comes back, and the CUDA row must
    not report the CPU run's step as its own.
    """

    _legacy_state(runtime_dir, BEAT_CUDA, fake_julia)
    monkeypatch.setattr(
        runtime, "_julia_gpu_functional", lambda julia, backend: (True, "probe says yes")
    )

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(provision, "_run_julia_step", interrupted)
    with pytest.raises(KeyboardInterrupt):
        provision.provision_cpu(runtime_dir, status_cb=lambda _: None)

    assert provision.read_state(runtime_dir, backend=BEAT_CPU)["status"] == "in_progress"
    assert provision.backend_ready(BEAT_CUDA) is True
    cuda = runtime.backend_status(BEAT_CUDA)
    assert cuda["available"] is True
    assert "in progress" not in cuda["reason"]


def test_a_failed_gpu_run_does_not_lose_a_legacy_cpu_record(
    runtime_dir, cuda_host, fake_julia, monkeypatch
):
    """The other order: legacy CPU readiness, then a GPU attempt that fails."""

    _legacy_state(runtime_dir, BEAT_CPU, fake_julia)

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA artifacts could not be resolved")

    monkeypatch.setattr(provision, "_run_julia_step", boom)

    assert provision.provision_gpu(
        runtime_dir, backend=BEAT_CUDA, status_cb=lambda _: None
    )["status"] == "failed"

    assert provision.backend_ready(BEAT_CPU) is True
    assert provision.provisioned_julia(runtime_dir, backend=BEAT_CPU) == str(fake_julia)
    monkeypatch.setattr(
        runtime, "_julia_gpu_functional", lambda julia, backend: (False, "still broken")
    )
    assert runtime.backend_status(BEAT_CPU)["available"] is True


def test_the_migration_leaves_an_existing_per_backend_record_alone(
    runtime_dir, no_gpu, fake_julia, julia_steps
):
    """A newer per-backend record outranks a legacy mirror describing it.

    The mirror is a copy of some earlier write, so treating it as an update
    would resurrect a stale verdict -- a ready record for a package that has
    since changed, for instance.
    """

    provision.provision_cpu(runtime_dir, status_cb=lambda _: None)
    provision._write_state(
        runtime_dir,
        {"status": "failed", "backend": BEAT_CPU, "error": "later attempt failed"},
    )
    # A hand-written legacy mirror claiming the opposite must not win.
    _legacy_state(runtime_dir, BEAT_CPU, fake_julia)
    provision._preserve_legacy_record(runtime_dir)

    assert provision.read_state(runtime_dir, backend=BEAT_CPU)["status"] == "failed"


def test_failed_migration_does_not_overwrite_the_only_legacy_record(
    runtime_dir, fake_julia, monkeypatch
):
    _legacy_state(runtime_dir, BEAT_CUDA, fake_julia)
    legacy_path = runtime_dir / provision.STATE_FILENAME
    before = legacy_path.read_bytes()
    replace = os.replace

    def refuse_cuda_migration(source, target):
        if Path(target) == provision.backend_state_path(runtime_dir, BEAT_CUDA):
            raise OSError(errno.ENOSPC, "no space to preserve CUDA readiness")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", refuse_cuda_migration)
    with pytest.raises(OSError, match="preserve CUDA readiness"):
        provision._write_state(runtime_dir, {"status": "in_progress", "backend": BEAT_CPU})
    assert legacy_path.read_bytes() == before
    assert provision.read_state(runtime_dir, backend=BEAT_CUDA)["status"] == "ready"
    assert not provision.backend_state_path(runtime_dir, BEAT_CPU).exists()


@pytest.mark.parametrize("error", [errno.EAGAIN, errno.EINVAL])
def test_only_lock_contention_is_retried(monkeypatch, error):
    if os.name == "nt":
        import msvcrt as locking_module
        function_name = "locking"
    else:
        import fcntl as locking_module
        function_name = "flock"

    def fail(*args):
        raise OSError(error, "lock operation failed")

    monkeypatch.setattr(locking_module, function_name, fail)
    if error == errno.EAGAIN:
        assert provision._try_lock(123) is False
    else:
        with pytest.raises(OSError, match="lock operation failed"):
            provision._try_lock(123)


# ---------------------------------------------------------------------------
# Capability reporting: per backend, and attributed to the right one.
# ---------------------------------------------------------------------------


def test_a_cpu_provisioning_in_progress_is_not_reported_as_a_cuda_one(
    runtime_dir, cuda_host, fake_julia, monkeypatch
):
    """The in-progress message used to read whichever backend was asked about.

    It came from the single record, so a CPU provisioning on a CUDA box made
    the CUDA row say "BEAT CUDA runtime provisioning is in progress" -- a
    sentence about work nobody started, on a row that was in fact ready.
    """

    monkeypatch.setattr(
        runtime, "_julia_gpu_functional", lambda julia, backend: (True, "probe says yes")
    )
    provision._write_state(
        runtime_dir,
        {"status": "in_progress", "backend": BEAT_CPU, "step": "instantiate"},
    )

    cuda = runtime.backend_status(BEAT_CUDA)
    assert cuda["available"] is True
    assert "in progress" not in cuda["reason"]

    cpu = runtime.backend_status(BEAT_CPU)
    assert cpu["available"] is False
    assert cpu["state"] == "provisioning"
    assert "instantiate" in cpu["reason"]


def test_two_accelerator_families_each_get_their_own_answer(
    runtime_dir, fake_julia, monkeypatch
):
    """One box, two cards. The single-answer probe could only name one.

    ``beat_engine_status`` still names one -- that is its question -- but a
    selector asking per backend now gets the truth about both.
    """

    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setattr(runtime, "_rocm_present", lambda: True)
    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: False)
    monkeypatch.setattr(
        runtime,
        "_julia_gpu_functional",
        lambda julia, backend: (
            (True, "CUDA.functional() confirmed a usable device")
            if backend == BEAT_CUDA
            else (False, "AMDGPU.functional() is false (no ROCm driver)")
        ),
    )

    statuses = runtime.beat_backend_statuses()

    assert statuses[BEAT_CUDA]["available"] is True
    assert statuses[BEAT_ROCM]["available"] is False
    assert statuses[BEAT_ROCM]["state"] == "not-functional"
    assert "AMDGPU.functional() is false" in statuses[BEAT_ROCM]["reason"]
    # An absent family says so plainly rather than borrowing another's verdict.
    assert statuses[BEAT_METAL]["state"] == "no-hardware"
    assert "Apple Silicon" in statuses[BEAT_METAL]["reason"]


def test_the_cpu_row_reports_the_provisioning_record_not_the_presence_of_julia(
    runtime_dir, cuda_host, fake_julia, julia_steps, monkeypatch
):
    monkeypatch.setattr(
        runtime, "_julia_gpu_functional", lambda julia, backend: (True, "probe says yes")
    )

    unprovisioned = runtime.backend_status(BEAT_CPU)
    assert unprovisioned["available"] is False
    assert unprovisioned["state"] == "unprovisioned"
    # And the remedy it names must not read as a trade against the GPU row.
    assert "does not disturb a provisioned GPU runtime" in unprovisioned["reason"]

    provision.provision_cpu(runtime_dir, status_cb=lambda _: None)
    assert runtime.backend_status(BEAT_CPU)["available"] is True
    assert runtime.backend_status(BEAT_CUDA)["available"] is True


def test_a_recorded_cpu_failure_is_reported_with_its_own_retry(
    runtime_dir, no_gpu, fake_julia, monkeypatch
):
    provision._write_state(
        runtime_dir,
        {
            "status": "failed",
            "backend": BEAT_CPU,
            "project": str(runtime.default_project(BEAT_CPU)),
            "error": "Not enough free disk space for the CPU runtime",
        },
    )

    status = runtime.backend_status(BEAT_CPU)

    assert status["state"] == "failed"
    assert "Not enough free disk space" in status["reason"]
    assert "--backend cpu --force" in status["reason"]


def test_a_broken_accelerator_no_longer_hides_a_provisioned_cpu_runtime(
    runtime_dir, cuda_host, fake_julia, julia_steps, monkeypatch
):
    """``beat_engine_status`` answers "what would a solve use", so answer it.

    A host with a present-but-unusable CUDA install and a provisioned CPU
    runtime can solve. Reporting BEAT unavailable there sent the whole engine
    family away over one broken driver.
    """

    monkeypatch.setattr(
        runtime,
        "_julia_gpu_functional",
        lambda julia, backend: (False, "CUDA.functional() is false (driver too old)"),
    )
    provision.provision_cpu(runtime_dir, status_cb=lambda _: None)

    status = runtime.beat_engine_status()

    assert status["available"] is True
    assert status["backend"] == BEAT_CPU
    assert "1 kHz solve" in status["reason"]


def test_an_unprovisioned_gpu_less_host_says_both_halves_of_why(
    runtime_dir, no_gpu, fake_julia
):
    status = runtime.beat_engine_status()

    assert status["available"] is False
    assert status["backend"] is None
    assert "No supported GPU was detected" in status["reason"]
    assert "not provisioned here" in status["reason"]
    assert runtime.provision_command(BEAT_CPU) in status["reason"]


def test_forcing_the_cpu_still_short_circuits_every_probe(
    runtime_dir, cuda_host, fake_julia, monkeypatch
):
    """CI has no provisioning record and must still exercise the plumbing."""

    monkeypatch.setenv(runtime.FORCE_CPU_ENV_VAR, "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("a Julia GPU probe ran although the CPU was forced")

    monkeypatch.setattr(runtime, "_julia_gpu_functional", forbidden)

    status = runtime.beat_engine_status()
    assert status["available"] is True
    assert status["backend"] == BEAT_CPU
    assert runtime.FORCE_CPU_ENV_VAR in status["reason"]
    assert runtime.backend_status(BEAT_CPU)["state"] == "forced"


def test_an_unknown_backend_is_a_failure_rather_than_an_empty_verdict(runtime_dir):
    with pytest.raises(ValueError, match="unknown BEAT backend"):
        runtime.backend_status("opencl")


def test_no_julia_is_reported_on_every_backend_row(runtime_dir, cuda_host, monkeypatch):
    monkeypatch.setattr(runtime, "discover_julia", lambda explicit=None: None)

    statuses = runtime.beat_backend_statuses()

    assert {name: item["state"] for name, item in statuses.items()} == {
        name: "no-julia" for name in statuses
    }
    for name, item in statuses.items():
        assert item["backend"] == name
        assert item["available"] is False


def test_migration_ignores_unknown_backend_names(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps({"backend": "invalid/../../escaped", "status": "ready"})
    )
    provision._write_state(runtime_dir, {"backend": BEAT_CPU, "status": "in_progress"})
    assert provision.read_state(runtime_dir, backend=BEAT_CPU)["status"] == "in_progress"


def test_interrupted_metal_record_does_not_hide_a_functional_runtime(runtime_dir, fake_julia, monkeypatch):
    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: True)
    monkeypatch.setattr(runtime, "_julia_gpu_functional", lambda julia, backend: (True, "functional"))
    provision._write_state(runtime_dir, {"backend": BEAT_METAL, "status": "in_progress"})
    assert runtime.backend_status(BEAT_METAL)["available"] is True
