"""Lifecycle of a Julia worker that outlives its client.

Every test here drives the production client and the production host; only
the Julia executable is replaced, by ``fake_julia_worker.py``, which speaks
the same JSON-lines protocol in a fraction of a second. That substitution is
deliberate and bounded: what these tests are about is process lifetime,
adoption, ownership and crash recovery, none of which involve the solver, and
paying a real cold start per test would make the suite both slow and
Julia-dependent. The real runtime is covered by ``test_solve_smoke.py``.

The properties asserted, in the order they matter:

1. a second *process* adopts the first one's runtime -- the whole point;
2. a worker that does not match the key is never adopted;
3. every way a host can be gone or wedged ends in a fresh, working one;
4. two processes racing to start one produce exactly one;
5. an abandoned solve does not poison the next client's worker;
6. an idle host retires itself, so orphans cannot accumulate.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from hornlab_beat_bem import worker_registry as registry
from hornlab_beat_bem.worker_client import (
    HostedBeatWorker,
    WorkerHandshakeError,
    find_live_hosts,
    keyed_environment,
)

FAKE_WORKER = Path(__file__).resolve().parent / "fake_julia_worker.py"
PROBE = Path(__file__).resolve().parent / "_worker_probe.py"

# Both transports are production paths: Unix sockets everywhere they fit, and
# loopback TCP on Windows and for any registry path too long for sun_path.
TRANSPORTS = ["unix", "tcp"] if os.name == "posix" else ["tcp"]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def registry_dir(isolated_worker_registry, monkeypatch):
    directory = Path(tempfile.mkdtemp(prefix="r-", dir=isolated_worker_registry))
    monkeypatch.setenv(registry.WORKER_DIR_ENV_VAR, str(directory))
    yield directory
    for record in find_live_hosts(directory):
        registry.terminate_pid(record.pid)


@pytest.fixture(params=TRANSPORTS)
def transport(request, monkeypatch):
    monkeypatch.setenv(registry.TRANSPORT_ENV_VAR, request.param)
    return request.param


def fake_key(
    *, threads: str = "1", version: str = "test-0.0.0", fingerprint: str = "fp0"
) -> dict:
    """A worker key that runs the fake worker instead of Julia."""

    return registry.worker_key(
        julia_executable=sys.executable,
        solver_script=FAKE_WORKER,
        julia_project=None,
        julia_sysimage=None,
        julia_threads=threads,
        package_version=version,
        package_fingerprint=fingerprint,
        environment={},
    )


def request_file(directory: Path, frequencies: list[float]) -> Path:
    path = directory / "request.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "config": {"tag_throat": 2},
                "cancel_path": str(directory / "cancel"),
                "frequencies_hz": frequencies,
            }
        ),
        encoding="utf-8",
    )
    return path


def run_probe(key: dict, directory: Path, action: str = "adopt", **kwargs) -> dict:
    key_path = directory / f"probe-{action}-{os.getpid()}-{time.time_ns()}.json"
    key_path.write_text(json.dumps(key), encoding="utf-8")
    environment = os.environ.copy()
    environment[registry.WORKER_DIR_ENV_VAR] = str(directory)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, str(PROBE), str(key_path), action],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
        **kwargs,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def wait_until(predicate, timeout: float = 15.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# --------------------------------------------------------------------------
# the key
# --------------------------------------------------------------------------


def test_the_key_separates_workers_that_must_not_be_shared():
    """Each field is in the key because sharing across it returns wrong work.

    The version, the fingerprint and the environment are the ones that were
    not in the old in-process key and had to be added: within one process they
    could not differ, and across an adoption they can. A checkout that moved
    between two app launches is the everyday case.
    """

    base = fake_key()
    assert registry.key_id(base) == registry.key_id(fake_key())
    assert registry.key_id(base) != registry.key_id(fake_key(threads="2"))
    assert registry.key_id(base) != registry.key_id(fake_key(version="test-0.0.1"))
    assert registry.key_id(base) != registry.key_id(fake_key(fingerprint="fp1"))

    with_runtime = dict(base, julia_executable="/different/julia")
    with_project = dict(base, julia_project="/different/project")
    with_sysimage = dict(base, julia_sysimage="/different/sysimage")
    assert registry.key_id(base) != registry.key_id(with_runtime)
    assert registry.key_id(base) != registry.key_id(with_project)
    assert registry.key_id(base) != registry.key_id(with_sysimage)

    with_env = dict(base)
    with_env["environment"] = {"BLAB_METAL_PIPELINE": "1"}
    assert registry.key_id(base) != registry.key_id(with_env)

    with_protocol = dict(base)
    with_protocol["protocol"] = registry.PROTOCOL_VERSION + 1
    assert registry.key_id(base) != registry.key_id(with_protocol)


def test_the_keyed_environment_is_the_julia_relevant_part_only(monkeypatch):
    monkeypatch.setenv("BLAB_METAL_PIPELINE", "1")
    monkeypatch.setenv("JULIA_DEPOT_PATH", "/depot")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    keyed = keyed_environment()
    assert keyed["BLAB_METAL_PIPELINE"] == "1"
    assert keyed["JULIA_DEPOT_PATH"] == "/depot"
    assert "PATH" not in keyed


def test_the_fingerprint_follows_the_code_not_the_version_number():
    """The version cannot carry staleness here, so the content has to.

    Consumers pin this repository by SHA, so ``package_version()`` is constant
    across every commit in a release cycle -- including a re-vendor of the
    engine. A key that trusted it would let a worker started before the change
    answer requests made after it.
    """

    from hornlab_beat_bem.runtime import PACKAGE_DIR, package_fingerprint

    before = package_fingerprint()
    assert before and before == package_fingerprint()  # cached, stable

    scratch = PACKAGE_DIR / "_fingerprint_probe.py"
    scratch.write_text("# temporary\n", encoding="utf-8")
    try:
        package_fingerprint.cache_clear()
        assert package_fingerprint() != before
    finally:
        scratch.unlink()
        package_fingerprint.cache_clear()
    assert package_fingerprint() == before


def test_the_fingerprint_includes_dependency_manifests_and_relative_names(
    monkeypatch, tmp_path
):
    from hornlab_beat_bem import runtime

    package = tmp_path / "package"
    backend = package / "julia_metal"
    bundle = package / "julia_engine" / "BeatEngineMetalBundle"
    backend.mkdir(parents=True)
    bundle.mkdir(parents=True)
    (package / "wrapper.py").write_text("# wrapper\n", encoding="utf-8")
    (backend / "Project.toml").write_text("name = \"Backend\"\n", encoding="utf-8")
    manifest = backend / "Manifest.toml"
    manifest.write_text("version = \"1\"\n", encoding="utf-8")
    bundle_project = bundle / "Project.toml"
    bundle_project.write_text("name = \"Bundle\"\n", encoding="utf-8")

    monkeypatch.setattr(runtime, "PACKAGE_DIR", package)
    runtime.package_fingerprint.cache_clear()
    try:
        before = runtime.package_fingerprint()
        os.utime(manifest, (1, 1))
        runtime.package_fingerprint.cache_clear()
        assert runtime.package_fingerprint() == before

        manifest.write_text("version = \"2\"\n", encoding="utf-8")
        runtime.package_fingerprint.cache_clear()
        after_manifest = runtime.package_fingerprint()
        assert after_manifest != before

        bundle_project.write_text("name = \"ChangedBundle\"\n", encoding="utf-8")
        runtime.package_fingerprint.cache_clear()
        before_rename = runtime.package_fingerprint()
        assert before_rename != after_manifest

        renamed = package / "julia_rocm"
        backend.rename(renamed)
        runtime.package_fingerprint.cache_clear()
        assert runtime.package_fingerprint() != before_rename
    finally:
        runtime.package_fingerprint.cache_clear()


def test_the_fingerprint_includes_custom_project_and_sysimage(tmp_path):
    from hornlab_beat_bem.runtime import package_fingerprint

    project = tmp_path / "custom-project"
    project.mkdir()
    (project / "Project.toml").write_text("name = \"Custom\"\n", encoding="utf-8")
    manifest = project / "Manifest.toml"
    manifest.write_text("version = \"1\"\n", encoding="utf-8")
    sysimage = tmp_path / "custom.so"
    sysimage.write_bytes(b"image-1")

    package_fingerprint.cache_clear()
    before = package_fingerprint(project, sysimage)
    os.utime(manifest, (1, 1))
    os.utime(sysimage, (1, 1))
    package_fingerprint.cache_clear()
    assert package_fingerprint(project, sysimage) == before

    manifest.write_text("version = \"2\"\n", encoding="utf-8")
    package_fingerprint.cache_clear()
    after_project = package_fingerprint(project, sysimage)
    assert after_project != before

    package_fingerprint.cache_clear()
    assert package_fingerprint(project / "Project.toml", sysimage) == after_project
    manifest.write_text("version = \"3\"\n", encoding="utf-8")
    package_fingerprint.cache_clear()
    after_project_file = package_fingerprint(project / "Project.toml", sysimage)
    assert after_project_file != after_project

    sysimage.write_bytes(b"image-2")
    package_fingerprint.cache_clear()
    assert package_fingerprint(project / "Project.toml", sysimage) != after_project_file


def test_worker_identity_uses_selected_project_and_sysimage_content(tmp_path):
    from hornlab_beat_bem.runtime import package_fingerprint
    from hornlab_beat_bem.worker import worker_key

    project = tmp_path / "custom-project"
    project.mkdir()
    (project / "Project.toml").write_text("name = \"Custom\"\n", encoding="utf-8")
    manifest = project / "Manifest.toml"
    manifest.write_text("version = \"1\"\n", encoding="utf-8")
    sysimage = tmp_path / "custom.so"
    sysimage.write_bytes(b"image-1")
    arguments = {
        "julia_executable": sys.executable,
        "solver_script": FAKE_WORKER,
        "julia_threads": "1",
        "julia_project": project,
        "julia_sysimage": sysimage,
    }

    package_fingerprint.cache_clear()
    before = worker_key(**arguments)
    manifest.write_text("version = \"2\"\n", encoding="utf-8")
    package_fingerprint.cache_clear()
    after_project = worker_key(**arguments)
    assert registry.key_id(after_project) != registry.key_id(before)

    sysimage.write_bytes(b"image-2")
    package_fingerprint.cache_clear()
    after_sysimage = worker_key(**arguments)
    assert registry.key_id(after_sysimage) != registry.key_id(after_project)


def test_windows_default_registry_is_stable_across_processes(tmp_path):
    """Source-level Windows check using two independent app processes."""

    environment = os.environ.copy()
    environment.pop(registry.WORKER_DIR_ENV_VAR, None)
    environment["LOCALAPPDATA"] = str(tmp_path / "local-app-data")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    command = [
        sys.executable,
        "-c",
        (
            "from hornlab_beat_bem.worker_registry import _default_worker_dir; "
            "print(_default_worker_dir('nt'))"
        ),
    ]

    first = subprocess.run(
        command, capture_output=True, text=True, check=True, env=environment
    ).stdout.strip()
    second = subprocess.run(
        command, capture_output=True, text=True, check=True, env=environment
    ).stdout.strip()
    assert first == second == str(
        tmp_path / "local-app-data" / "HornLab" / "BEAT" / "workers"
    )


@pytest.mark.parametrize("fallback", ["local_app_data", "home"])
def test_windows_default_registry_fallbacks_are_local_and_pid_independent(
    fallback, monkeypatch, tmp_path
):
    local_app_data = tmp_path / "local-app-data"
    home = tmp_path / "home"
    roaming = tmp_path / "roaming-must-not-be-used"
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    if fallback == "local_app_data":
        monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
        monkeypatch.setattr(
            Path, "home", lambda: (_ for _ in ()).throw(AssertionError("home used"))
        )
        expected_root = local_app_data
    elif fallback == "home":
        monkeypatch.setattr(Path, "home", lambda: home)
        expected_root = home / "AppData" / "Local"

    first = registry._default_worker_dir("nt")
    second = registry._default_worker_dir("nt")
    assert first == second == expected_root / "HornLab" / "BEAT" / "workers"
    assert roaming not in first.parents


def test_windows_default_registry_refuses_a_shared_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming-must-not-be-used"))
    monkeypatch.setattr(
        Path, "home", lambda: (_ for _ in ()).throw(RuntimeError("no home"))
    )

    with pytest.raises(RuntimeError, match=registry.WORKER_DIR_ENV_VAR):
        registry._default_worker_dir("nt")


def test_worker_dir_creates_the_selected_default(monkeypatch, tmp_path):
    expected = tmp_path / "registry"
    monkeypatch.delenv(registry.WORKER_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(registry, "_default_worker_dir", lambda: expected)

    assert registry.worker_dir() == expected
    assert expected.is_dir()


def test_a_socket_path_too_long_for_sun_path_selects_loopback(monkeypatch, tmp_path):
    monkeypatch.delenv(registry.TRANSPORT_ENV_VAR, raising=False)
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    endpoint = registry.endpoint_for("0123456789abcdef", deep)
    assert isinstance(endpoint, registry.TcpEndpoint)


# --------------------------------------------------------------------------
# adoption
# --------------------------------------------------------------------------


def test_a_second_process_adopts_the_running_worker(registry_dir, transport):
    """The headline behaviour: an app restart does not pay the cold start.

    Two separate interpreters, one after the other, exactly as two launches of
    the application would be. The engine pid is what is asserted, not the host
    pid: a host that quietly restarted its Julia child would look identical by
    host pid and would have thrown away every compiled kernel.
    """

    first = run_probe(fake_key(), registry_dir)
    second = run_probe(fake_key(), registry_dir)
    assert first["host_pid"] == second["host_pid"]
    assert first["engine_pid"] == second["engine_pid"]
    assert registry.pid_alive(first["host_pid"])


def test_a_different_key_never_adopts(registry_dir, transport):
    one = run_probe(fake_key(threads="1"), registry_dir)
    two = run_probe(fake_key(threads="2"), registry_dir)
    assert one["host_pid"] != two["host_pid"]
    assert len(find_live_hosts(registry_dir)) == 2


def test_the_host_refuses_a_client_that_fails_the_handshake(registry_dir, transport):
    """Identity is checked by the host, not only by the client.

    The client's own check reads a file it could have been handed; this is the
    one a wrong client cannot talk its way past.
    """

    key = fake_key()
    worker = HostedBeatWorker(key, directory=registry_dir)
    worker.ensure_started()
    record = registry.read_key_file(registry.key_id(key), registry_dir)
    assert record is not None

    for bad in (
        {"token": "not-the-token"},
        {"key": dict(key, julia_threads="99")},
        {"protocol": registry.PROTOCOL_VERSION + 1},
    ):
        connection = record.endpoint.connect(timeout=10.0)
        hello = {
            "op": "hello",
            "protocol": registry.PROTOCOL_VERSION,
            "key_id": registry.key_id(key),
            "key": key,
            "token": record.token,
        }
        hello.update(bad)
        registry.send_frame(connection, hello)
        reply = registry.receive_frame(connection)
        connection.close()
        assert reply is not None and reply["type"] == "hello_refused", bad

    # A client that reaches the handshake with the wrong credentials is told
    # so, rather than being handed a connection it would then drive.
    impostor = HostedBeatWorker(key, directory=registry_dir)
    with pytest.raises(WorkerHandshakeError):
        impostor._handshake(
            registry.KeyFile(
                identifier=record.identifier,
                key=record.key,
                pid=record.pid,
                token="not-the-token",
                endpoint=record.endpoint,
            )
        )

    # The rejections cost the legitimate client nothing.
    assert worker.ping()["host_pid"] == record.pid
    worker.shutdown()


def test_a_stale_record_whose_host_is_gone_is_replaced(registry_dir, transport):
    """A pid from a host that died without cleaning up after itself."""

    key = fake_key()
    identifier = registry.key_id(key)
    registry.write_key_file(
        registry.KeyFile(
            identifier=identifier,
            key=key,
            pid=_dead_pid(),
            token="stale",
            endpoint=registry.endpoint_for(identifier, registry_dir),
        ),
        registry_dir,
    )
    worker = HostedBeatWorker(key, directory=registry_dir)
    worker.ensure_started()
    assert worker.host_pid is not None and registry.pid_alive(worker.host_pid)
    assert registry.read_key_file(identifier, registry_dir).token != "stale"
    worker.shutdown()


def test_a_record_pointing_at_a_socket_nobody_answers_is_replaced(registry_dir):
    """The socket file outlives its host, so ``connect`` is the only witness.

    The recorded pid is this test's own -- alive, and therefore useless as a
    liveness signal. Only the refused connection distinguishes a host that is
    there from one that is not, which is why the client cannot stop at the
    record.
    """

    key = fake_key()
    identifier = registry.key_id(key)
    orphan = registry_dir / "orphan.sock"
    orphan.write_text("not a socket", encoding="utf-8")
    registry.write_key_file(
        registry.KeyFile(
            identifier=identifier,
            key=key,
            pid=os.getpid(),
            token="stale",
            endpoint=registry.UnixEndpoint(orphan),
        ),
        registry_dir,
    )
    worker = HostedBeatWorker(key, directory=registry_dir)
    worker.ensure_started()
    assert worker.host_pid not in (None, os.getpid())
    assert registry.pid_alive(worker.host_pid)
    worker.shutdown()


def test_concurrent_adoption_starts_exactly_one_host(registry_dir, transport):
    """Two applications launched together must not start two workers.

    Separate processes, started as close to simultaneously as the operating
    system allows, all racing the same empty registry -- the scenario the
    cross-process spawn lock exists for.
    """

    key = fake_key()
    key_path = registry_dir / "race-key.json"
    key_path.write_text(json.dumps(key), encoding="utf-8")
    environment = os.environ.copy()
    environment[registry.WORKER_DIR_ENV_VAR] = str(registry_dir)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    processes = [
        subprocess.Popen(
            [sys.executable, str(PROBE), str(key_path), "adopt"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for _ in range(5)
    ]
    reports = []
    for process in processes:
        out, err = process.communicate(timeout=120)
        assert process.returncode == 0, err
        reports.append(json.loads(out.strip().splitlines()[-1]))

    host_pids = {report["host_pid"] for report in reports}
    assert len(host_pids) == 1, f"the race started {len(host_pids)} hosts: {host_pids}"
    assert len({report["engine_pid"] for report in reports}) == 1
    assert len(find_live_hosts(registry_dir)) == 1


# --------------------------------------------------------------------------
# ownership and abandonment
# --------------------------------------------------------------------------


def test_one_job_at_a_time_across_processes(registry_dir, monkeypatch):
    """A second client is queued, not refused, and not run concurrently.

    Refusing would turn an ordinary overlap -- a warm-up and the first solve
    of a restarted app -- into an error with no sensible retry. Queuing is
    also what the in-process lock did, so nothing about the caller changes.
    """

    monkeypatch.setenv("FAKE_BEAT_SOLVE_S", "0.4")
    key = fake_key()
    worker = HostedBeatWorker(key, directory=registry_dir)
    worker.ensure_started()

    key_path = registry_dir / "queued-key.json"
    key_path.write_text(json.dumps(key), encoding="utf-8")
    environment = os.environ.copy()
    environment[registry.WORKER_DIR_ENV_VAR] = str(registry_dir)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        stream = worker.submit(request_file(directory, [100.0, 200.0, 300.0]))
        types = []
        while "initialized" not in types:
            types.append(str(next(stream).get("type")))
        started = time.monotonic()
        competitor = subprocess.Popen(
            [sys.executable, str(PROBE), str(key_path), "solve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        # Drain our own job; the competitor cannot have finished before it.
        types += [str(event.get("type")) for event in stream]
        assert types[-1] == "completed"
        out, err = competitor.communicate(timeout=120)
        assert competitor.returncode == 0, err
        report = json.loads(out.strip().splitlines()[-1])

    assert report["events"][-1] == "completed"
    assert report["host_pid"] == worker.host_pid
    # 3 + 2 frequencies at 0.4 s each: serialised, never overlapped.
    assert time.monotonic() - started >= 0.8
    worker.shutdown()


def test_a_client_that_dies_mid_solve_leaves_a_usable_worker(registry_dir, monkeypatch):
    """The crash case that would otherwise poison every later adoption.

    Without the host noticing the dead connection, Julia keeps solving into a
    pipe nobody reads and the next client's first events belong to the
    abandoned job.
    """

    monkeypatch.setenv("FAKE_BEAT_SOLVE_S", "0.3")
    key = fake_key()
    abandoned = run_probe(key, registry_dir, action="abandon")
    assert wait_until(lambda: registry.pid_alive(abandoned["host_pid"]))

    recovered = run_probe(key, registry_dir, action="solve")
    assert recovered["host_pid"] == abandoned["host_pid"], "the host itself must survive"
    assert recovered["events"][0] == "initialized"
    assert recovered["events"][-1] == "completed"
    # The runtime was retired and replaced, not handed over mid-job.
    assert recovered["engine_pid"] != abandoned["engine_pid"]


def test_retiring_the_engine_keeps_the_host_and_re_warms(registry_dir):
    """``terminate`` is a retirement, not a shutdown -- and it re-warms.

    This is the "re-warm after a cancel" behaviour: the replacement runtime
    starts immediately, so its start-up is paid while nobody is waiting
    rather than by the next solve.
    """

    worker = HostedBeatWorker(fake_key(), directory=registry_dir)
    worker.ensure_started()
    host_pid, first_engine = worker.host_pid, worker.engine_pid
    worker.terminate()
    assert registry.pid_alive(host_pid)

    worker.ensure_started()
    assert worker.host_pid == host_pid
    assert worker.engine_pid != first_engine
    worker.shutdown()


# --------------------------------------------------------------------------
# lifetime
# --------------------------------------------------------------------------


def test_detach_leaves_the_worker_running_and_shutdown_stops_it(registry_dir, transport):
    """The application contract, both halves of it."""

    key = fake_key()
    worker = HostedBeatWorker(key, directory=registry_dir)
    worker.ensure_started()
    host_pid = worker.host_pid

    worker.detach()
    assert registry.pid_alive(host_pid)
    assert registry.read_key_file(registry.key_id(key), registry_dir) is not None

    adopted = HostedBeatWorker(key, directory=registry_dir)
    adopted.ensure_started()
    assert adopted.host_pid == host_pid

    adopted.shutdown()
    assert wait_until(lambda: not registry.pid_alive(host_pid))
    assert registry.read_key_file(registry.key_id(key), registry_dir) is None


def test_an_idle_host_retires_itself(registry_dir, monkeypatch):
    """Orphans cannot accumulate, whatever an application forgets to do."""

    monkeypatch.setenv(registry.IDLE_TIMEOUT_ENV_VAR, "1")
    key = fake_key()
    worker = HostedBeatWorker(key, directory=registry_dir)
    worker.ensure_started()
    host_pid = worker.host_pid
    worker.detach()

    assert wait_until(lambda: not registry.pid_alive(host_pid), timeout=30.0)
    assert registry.read_key_file(registry.key_id(key), registry_dir) is None
    if os.name == "posix" and os.environ.get(registry.TRANSPORT_ENV_VAR) != "tcp":
        assert not (registry_dir / f"{registry.key_id(key)}.sock").exists()


def test_a_connected_client_holds_the_worker_open(registry_dir, monkeypatch):
    """The idle clock must not run while somebody is attached.

    An application can sit connected for an hour without solving; retiring
    the worker under it would reintroduce the cold start this whole mechanism
    exists to remove.
    """

    monkeypatch.setenv(registry.IDLE_TIMEOUT_ENV_VAR, "1")
    worker = HostedBeatWorker(fake_key(), directory=registry_dir)
    worker.ensure_started()
    host_pid = worker.host_pid
    time.sleep(2.5)
    assert registry.pid_alive(host_pid)
    assert worker.ping()["host_pid"] == host_pid
    worker.shutdown()


def test_a_failed_startup_is_reported_not_swallowed(registry_dir, monkeypatch):
    monkeypatch.setenv("FAKE_BEAT_FAIL", "1")
    worker = HostedBeatWorker(fake_key(), directory=registry_dir)
    with pytest.raises(RuntimeError):
        worker.ensure_started()
    worker.shutdown()


# --------------------------------------------------------------------------
# integration with the public surface
# --------------------------------------------------------------------------


def test_get_worker_returns_a_hosted_worker_and_can_be_switched_off(
    registry_dir, monkeypatch
):
    from hornlab_beat_bem.worker import (
        BeatWorkerProcess,
        detach_workers,
        get_worker,
        shutdown_workers,
    )
    from hornlab_beat_bem.worker_client import HostedBeatWorker as Hosted

    hosted = get_worker(
        julia_executable=sys.executable,
        solver_script=FAKE_WORKER,
        julia_threads="1",
        julia_project=None,
    )
    assert isinstance(hosted, Hosted)
    hosted.ensure_started()
    host_pid = hosted.host_pid

    detach_workers()
    assert registry.pid_alive(host_pid)

    # A second get_worker in this process adopts the same host, exactly as a
    # second process would.
    again = get_worker(
        julia_executable=sys.executable,
        solver_script=FAKE_WORKER,
        julia_threads="1",
        julia_project=None,
    )
    again.ensure_started()
    assert again.host_pid == host_pid
    shutdown_workers()
    assert wait_until(lambda: not registry.pid_alive(host_pid))

    monkeypatch.setenv(registry.PERSISTENT_HOST_ENV_VAR, "0")
    child = get_worker(
        julia_executable=sys.executable,
        solver_script=FAKE_WORKER,
        julia_threads="1",
        julia_project=None,
    )
    assert isinstance(child, BeatWorkerProcess)
    child.ensure_started()
    assert child.pid is not None
    shutdown_workers()


def test_solve_session_streams_through_the_host(registry_dir, monkeypatch):
    """``BeatSolveSession`` is unchanged by the transport underneath it."""

    from hornlab_beat_bem.worker import BeatSolveSession, shutdown_workers

    # The session fills in the backend's Julia project when none is given, and
    # ``--project=`` is not something the stand-in interpreter accepts.
    monkeypatch.setattr("hornlab_beat_bem.worker.default_project", lambda backend: None)
    statuses: list[str] = []
    session = BeatSolveSession(
        {"schema_version": 2, "config": {"tag_throat": 2}, "frequencies_hz": [100.0, 200.0]},
        julia_executable=sys.executable,
        beat_backend="cpu",
        solver_script=FAKE_WORKER,
        julia_project=None,
        julia_threads="1",
        status_callback=statuses.append,
    )
    assert session.metadata is not None
    assert session.metadata["polar_angle_deg"] == [0.0, 45.0, 90.0]
    events = [str(event.get("type")) for event in session.events()]
    assert events == ["result", "result", "completed"]
    shutdown_workers()


def _dead_pid() -> int:
    """A process id that certainly is not running.

    Preferably one recycled from a process that has been reaped, because that
    is the shape the client meets in the field. A pid can be reused before
    this returns, though, so the fallback is a value no pid table reaches --
    never a skip, since a skip in this repository is a lost check.
    """

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    pid = process.pid
    for _ in range(200):
        if not registry.pid_alive(pid):
            return pid
        time.sleep(0.01)
    return 0x7FFFFFFF


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal semantics")
def test_signalling_a_host_stops_it_cleanly(registry_dir):
    """A SIGTERM from a session cleanup must not leave a record behind."""

    key = fake_key()
    worker = HostedBeatWorker(key, directory=registry_dir)
    worker.ensure_started()
    host_pid = worker.host_pid
    os.kill(host_pid, signal.SIGTERM)
    assert wait_until(lambda: not registry.pid_alive(host_pid))
    assert registry.read_key_file(registry.key_id(key), registry_dir) is None

    # And the next client simply starts a new one.
    replacement = HostedBeatWorker(key, directory=registry_dir)
    replacement.ensure_started()
    assert replacement.host_pid != host_pid
    replacement.shutdown()
