"""Provisioning: GPU stays hardware-gated, CPU is an explicit opt-in.

The two halves are asserted against each other on purpose. The GPU tests say
no hardware means no download, ever; the CPU tests say an installer that asks
for the CPU by name gets one on any host -- and neither backend's *ready*
record is ever allowed to answer for the other.
"""

from __future__ import annotations

import json
import re
import tomllib

import pytest

from hornlab_beat_bem import provision, runtime
from hornlab_beat_bem.config import BEAT_CPU, BEAT_CUDA, BEAT_METAL, BEAT_ROCM
from hornlab_beat_bem.runtime import PACKAGE_DIR


@pytest.fixture()
def runtime_dir(tmp_path, monkeypatch):
    directory = tmp_path / "runtime"
    monkeypatch.setenv(provision.RUNTIME_DIR_ENV_VAR, str(directory))
    return directory


@pytest.mark.parametrize(
    ("backend", "hardware", "label"),
    [
        (BEAT_CUDA, "NVIDIA GPU", "CUDA"),
        (BEAT_ROCM, "AMD ROCm runtime", "ROCm"),
        (BEAT_METAL, "Apple Silicon GPU", "Metal"),
    ],
)
def test_no_gpu_means_no_download_and_no_state(
    runtime_dir, monkeypatch, backend, hardware, label
):
    monkeypatch.setattr(provision, "_gpu_hardware_present", lambda selected: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("provisioning attempted a download without a GPU")

    monkeypatch.setattr(provision, "_ensure_julia", forbidden)
    monkeypatch.setattr(provision, "_download", forbidden)

    messages = []
    state = provision.provision_gpu(runtime_dir, backend=backend, status_cb=messages.append)
    assert state["status"] == "skipped"
    assert state["reason"] == f"no {hardware} detected"
    assert messages == [
        f"No {hardware} detected; skipping BEAT {label} runtime provisioning."
    ]
    assert not runtime_dir.exists()


def test_cli_gate_exits_quietly_without_gpu(runtime_dir, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: False)
    assert provision.main(["--if-nvidia-gpu"]) == 0
    assert capsys.readouterr().out == ""
    assert not runtime_dir.exists()


def test_existing_julia_skips_the_julia_download(runtime_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: True)
    fake_julia = tmp_path / "julia.exe"
    fake_julia.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_julia", lambda explicit=None: str(fake_julia))

    def forbidden(*args, **kwargs):
        raise AssertionError("Julia download attempted although Julia exists")

    monkeypatch.setattr(provision, "_download", forbidden)
    steps: list[str] = []
    monkeypatch.setattr(
        provision,
        "_run_julia_step",
        lambda julia, code, **kwargs: steps.append(kwargs["label"]),
    )

    state = provision.provision_cuda(runtime_dir, status_cb=lambda _: None)
    assert state["status"] == "ready"
    assert state["julia_executable"] == str(fake_julia)
    assert len(steps) == 2  # instantiate + GPU probe
    assert provision.provisioned_julia(runtime_dir) == str(fake_julia)


def test_backend_gating_matches_hardware(runtime_dir, monkeypatch, tmp_path):
    """A ROCm-only host provisions ROCm and skips CUDA, and vice versa."""

    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setattr(runtime, "_rocm_present", lambda: True)
    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: False)
    fake_julia = tmp_path / "julia.exe"
    fake_julia.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_julia", lambda explicit=None: str(fake_julia))

    assert provision.detect_gpu_backend() == "rocm"
    cuda_state = provision.provision_gpu(
        runtime_dir, backend="cuda", status_cb=lambda _: None
    )
    assert cuda_state["status"] == "skipped"

    probes: list[str] = []
    monkeypatch.setattr(
        provision,
        "_run_julia_step",
        lambda julia, code, **kwargs: probes.append(str(kwargs["project"])),
    )
    rocm_state = provision.provision_gpu(
        runtime_dir, backend="rocm", status_cb=lambda _: None
    )
    assert rocm_state["status"] == "ready"
    assert rocm_state["backend"] == "rocm"
    assert all(path.endswith("julia_rocm") for path in probes)


def test_cli_if_gpu_gate_exits_quietly_without_any_gpu(runtime_dir, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setattr(runtime, "_rocm_present", lambda: False)
    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: False)
    assert provision.main(["--if-gpu"]) == 0
    assert capsys.readouterr().out == ""
    assert not runtime_dir.exists()


def test_ready_state_short_circuits(runtime_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: True)
    fake_julia = tmp_path / "julia.exe"
    fake_julia.write_text("", encoding="utf-8")
    runtime_dir.mkdir(parents=True)
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps(
            {
                "status": "ready",
                "backend": "cuda",
                "project": str(runtime.default_project("cuda")),
                "package_fingerprint": runtime.package_fingerprint(
                    runtime.default_project("cuda")
                ),
                "julia_executable": str(fake_julia),
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("re-provisioned although state is ready")

    monkeypatch.setattr(provision, "_ensure_julia", forbidden)
    monkeypatch.setattr(provision, "_run_julia_step", forbidden)
    state = provision.provision_cuda(runtime_dir, status_cb=lambda _: None)
    assert state["status"] == "ready"


def test_failure_is_recorded_not_raised(runtime_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: True)
    fake_julia = tmp_path / "julia.exe"
    fake_julia.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_julia", lambda explicit=None: str(fake_julia))

    def boom(*args, **kwargs):
        raise RuntimeError("instantiate exploded")

    monkeypatch.setattr(provision, "_run_julia_step", boom)
    state = provision.provision_cuda(runtime_dir, status_cb=lambda _: None)
    assert state["status"] == "failed"
    assert "instantiate exploded" in state["error"]
    stored = provision.read_state(runtime_dir)
    assert stored is not None and stored["status"] == "failed"
    assert provision.provisioned_julia(runtime_dir) is None


def test_download_rejects_sha_mismatch(tmp_path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"julia bits")
    destination = tmp_path / "out" / "artifact.bin"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        provision._download(
            source.as_uri(),
            destination,
            expected_sha256="0" * 64,
            status_cb=lambda _: None,
        )
    assert not destination.exists()


def test_discover_julia_prefers_env_then_provisioned(monkeypatch, tmp_path):
    provisioned = tmp_path / "prov" / "julia.exe"
    provisioned.parent.mkdir()
    provisioned.write_text("", encoding="utf-8")
    monkeypatch.setattr(provision, "provisioned_julia", lambda runtime_dir=None: str(provisioned))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: r"C:\path\julia.exe")
    monkeypatch.delenv(runtime.JULIA_ENV_VAR, raising=False)
    assert runtime.discover_julia() == str(provisioned)

    env_julia = tmp_path / "env" / "julia.exe"
    env_julia.parent.mkdir()
    env_julia.write_text("", encoding="utf-8")
    monkeypatch.setenv(runtime.JULIA_ENV_VAR, str(env_julia))
    assert runtime.discover_julia() == str(env_julia)


def test_metal_host_provisions_metal_and_skips_the_others(runtime_dir, monkeypatch, tmp_path):
    """Apple Silicon is a GPU host like any other: it provisions julia_metal."""

    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setattr(runtime, "_rocm_present", lambda: False)
    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: True)
    fake_julia = tmp_path / "julia"
    fake_julia.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_julia", lambda explicit=None: str(fake_julia))

    assert provision.detect_gpu_backend() == "metal"
    assert provision.provision_gpu(
        runtime_dir, backend="cuda", status_cb=lambda _: None
    )["status"] == "skipped"

    probes: list[str] = []
    monkeypatch.setattr(
        provision,
        "_run_julia_step",
        lambda julia, code, **kwargs: probes.append(str(kwargs["project"])),
    )
    state = provision.provision_gpu(runtime_dir, backend="metal", status_cb=lambda _: None)
    assert state["status"] == "ready"
    assert state["backend"] == "metal"
    assert all(path.endswith("julia_metal") for path in probes)


@pytest.fixture()
def no_gpu(monkeypatch):
    """The host the CPU path exists for: nothing accelerated anywhere."""

    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setattr(runtime, "_rocm_present", lambda: False)
    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: False)


@pytest.fixture()
def fake_julia(monkeypatch, tmp_path):
    """A Julia that discovery finds, so no test here can reach the network."""

    executable = tmp_path / "julia.exe"
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_julia", lambda explicit=None: str(executable))

    def forbidden(*args, **kwargs):
        raise AssertionError("a download was attempted although Julia exists")

    monkeypatch.setattr(provision, "_download", forbidden)
    return executable


@pytest.fixture()
def julia_steps(monkeypatch):
    """Record every Julia invocation instead of running one."""

    calls: list[dict] = []

    def record(julia, code, **kwargs):
        calls.append({"julia": julia, "code": code, **kwargs})

    monkeypatch.setattr(provision, "_run_julia_step", record)
    return calls


def test_cpu_provisioning_needs_no_gpu_at_all(runtime_dir, no_gpu, fake_julia, julia_steps):
    """The whole point: a GPU-less host can opt in and get a runtime."""

    state = provision.provision_cpu(runtime_dir, status_cb=lambda _: None)

    assert state["status"] == "ready"
    assert state["backend"] == BEAT_CPU
    assert state["julia_executable"] == str(fake_julia)
    # Discovery's third tier is what this exists to make live.
    assert provision.provisioned_julia(runtime_dir) == str(fake_julia)
    assert [call["env_backend"] for call in julia_steps] == [BEAT_CPU, BEAT_CPU]
    assert all(str(call["project"]).endswith("julia") for call in julia_steps)


def test_a_raising_status_callback_cannot_fail_provisioning(
    runtime_dir, no_gpu, fake_julia, julia_steps, capsys
):
    """A console that cannot print a status line must not fail the runtime.

    On a Windows cp1252 console ``print`` raises ``UnicodeEncodeError`` for
    Pkg's check-mark lines; that used to record the run as failed.
    """

    seen = []

    def cp1252_console(message):
        seen.append(message)
        raise UnicodeEncodeError(
            "cp1252", "\u2713", 0, 1, "character maps to <undefined>"
        )

    state = provision.provision_cpu(runtime_dir, status_cb=cp1252_console)

    assert state["status"] == "ready"
    assert seen, "the callback was still offered every status line"
    assert provision.read_state(runtime_dir, backend=BEAT_CPU)["status"] == "ready"
    err = capsys.readouterr().err
    assert err.count("status callback failed (UnicodeEncodeError)") == 1


def test_guarded_status_passes_lines_through_and_wraps_once():
    lines = []
    guarded = provision._guarded_status(lines.append)
    guarded("one")
    assert lines == ["one"]
    assert provision._guarded_status(guarded) is guarded


def test_cpu_instantiate_pulls_no_gpu_artifacts(runtime_dir, no_gpu, fake_julia, julia_steps):
    """The CPU project must be the CPU project, not a GPU one with a label."""

    provision.provision_cpu(runtime_dir, status_cb=lambda _: None)

    projects = {str(call["project"]) for call in julia_steps}
    assert projects == {str(PACKAGE_DIR / "julia")}
    project = tomllib.loads((PACKAGE_DIR / "julia" / "Project.toml").read_text(encoding="utf-8"))
    manifest = tomllib.loads(
        (PACKAGE_DIR / "julia" / "Manifest.toml").read_text(encoding="utf-8")
    )
    accelerator_packages = {"CUDA", "AMDGPU", "Metal"}
    assert accelerator_packages.isdisjoint(project.get("deps", {}))
    assert accelerator_packages.isdisjoint(manifest.get("deps", {}))
    assert not any(
        name in str(call["code"]) for call in julia_steps for name in ("CUDA", "AMDGPU", "Metal")
    )
    assert not any("several GB" in str(call["label"]) for call in julia_steps)


def test_cpu_probe_solves_through_the_bundle(runtime_dir, no_gpu, fake_julia, julia_steps):
    """Instantiating is not evidence; the probe has to reach a real solve.

    A bundle that is missing, or that resolved a foreign engine directory,
    still lets ``solver.jl`` compile from source and answer correctly -- three
    to five times slower, with nothing logged (AGENTS.md). So *ready* is only
    honest if something solved.
    """

    provision.provision_cpu(runtime_dir, status_cb=lambda _: None)

    probe = julia_steps[-1]
    code = str(probe["code"])
    assert "using BeatEngineCpuBundle" in code
    assert "solve_request(request)" in code
    assert "B.ENGINE_DIR" in code
    assert "samefile(found, expected)" in code
    # The probe compares against this package's engine directory, passed by
    # environment so no path has to survive Julia string quoting on Windows.
    assert probe["extra_env"] == {
        provision._CPU_PROBE_ENGINE_DIR_ENV_VAR: str(PACKAGE_DIR / "julia")
    }
    assert '"beat_engine_backend" => "cpu"' in code
    assert 'pressure["imag"]' in code
    assert 'event_type == "completed"' in code


def test_cpu_probe_asks_for_what_the_bundle_workload_asks_for():
    """The probe's request must not drift away from the vendored workload's.

    Both build a solve request by hand against the same driver. The workload
    swallows its own failures on purpose (a build must not fail over an
    optimisation), so if a re-sync adds a required config key the workload is
    corrected and the probe silently starts sending an incomplete request --
    which would turn *ready* into "the probe reliably fails".
    """

    bundle = (
        PACKAGE_DIR
        / "julia_engine"
        / "BeatEngineCpuBundle"
        / "src"
        / "BeatEngineCpuBundle.jl"
    ).read_text(encoding="utf-8")
    keys = re.compile(r'"(\w+)" =>')
    workload = set(keys.findall(bundle.split("WORKLOAD_MESH)", 1)[1].split("redirect_stdout", 1)[0]))
    probe = set(keys.findall(provision._CPU_PROBE_CODE))
    assert workload, "the bundle workload no longer builds a request inline"
    assert workload <= probe, f"the probe request is missing {sorted(workload - probe)}"


def test_cpu_probe_computes_its_verdict_where_julia_can_see_it():
    """Julia's soft scope would quietly localise a loop's assignment.

    Written as top-level statements, ``results``/``usable`` are assigned inside
    a ``for`` loop and read outside it -- the exact shape the soft-scope rule
    turns into a local plus a warning, leaving the check reading initial
    values and passing whatever happened.
    """

    code = provision._CPU_PROBE_CODE
    assert "function beat_cpu_probe()" in code
    assert code.strip().endswith("exit(beat_cpu_probe())")
    body = code.split("function beat_cpu_probe()", 1)[1]
    assert "for line in eachline(events)" in body


def test_a_cpu_ready_runtime_does_not_satisfy_a_gpu_request(runtime_dir, fake_julia, julia_steps, monkeypatch):
    """A cached *ready* is a claim about one backend, never about both."""

    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: True)
    runtime_dir.mkdir(parents=True)
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps(
            {
                "status": "ready",
                "backend": BEAT_CPU,
                "project": str(PACKAGE_DIR / "julia"),
                "package_fingerprint": runtime.package_fingerprint(
                    PACKAGE_DIR / "julia"
                ),
                "julia_executable": str(fake_julia),
            }
        ),
        encoding="utf-8",
    )

    state = provision.provision_cuda(runtime_dir, status_cb=lambda _: None)

    assert state["backend"] == "cuda"
    assert julia_steps, "the CUDA request reused a CPU-ready record"
    assert all(str(call["project"]).endswith("julia_cuda") for call in julia_steps)


def test_a_gpu_ready_runtime_does_not_satisfy_a_cpu_request(runtime_dir, no_gpu, fake_julia, julia_steps):
    runtime_dir.mkdir(parents=True)
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps(
            {"status": "ready", "backend": "cuda", "julia_executable": str(fake_julia)}
        ),
        encoding="utf-8",
    )

    state = provision.provision_cpu(runtime_dir, status_cb=lambda _: None)

    assert state["backend"] == BEAT_CPU
    assert julia_steps, "the CPU request reused a CUDA-ready record"


def test_cpu_ready_state_short_circuits(runtime_dir, no_gpu, fake_julia, monkeypatch):
    runtime_dir.mkdir(parents=True)
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps(
            {
                "status": "ready",
                "backend": BEAT_CPU,
                "project": str(PACKAGE_DIR / "julia"),
                "package_fingerprint": runtime.package_fingerprint(
                    PACKAGE_DIR / "julia"
                ),
                "julia_executable": str(fake_julia),
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("re-provisioned although the CPU state is ready")

    monkeypatch.setattr(provision, "_run_julia_step", forbidden)
    assert provision.provision_cpu(runtime_dir, status_cb=lambda _: None)["status"] == "ready"


def test_ready_state_reprovisions_after_same_path_content_change(
    runtime_dir, no_gpu, fake_julia, julia_steps, monkeypatch, tmp_path
):
    project = tmp_path / "julia"
    project.mkdir()
    (project / "Project.toml").write_text("[deps]\n", encoding="utf-8")
    manifest = project / "Manifest.toml"
    manifest.write_text("manifest_format = \"2.0\"\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "default_project", lambda backend: project)
    runtime.package_fingerprint.cache_clear()
    fingerprint = runtime.package_fingerprint(project)
    runtime_dir.mkdir(parents=True)
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps(
            {
                "status": "ready",
                "backend": BEAT_CPU,
                "project": str(project),
                "package_fingerprint": fingerprint,
                "julia_executable": str(fake_julia),
            }
        ),
        encoding="utf-8",
    )

    assert provision.provision_cpu(runtime_dir, status_cb=lambda _: None)["status"] == "ready"
    assert not julia_steps

    manifest.write_text("manifest_format = \"2.0\"\nproject_hash = \"changed\"\n", encoding="utf-8")
    runtime.package_fingerprint.cache_clear()  # a package update starts a new process
    assert provision.provision_cpu(runtime_dir, status_cb=lambda _: None)["status"] == "ready"
    assert len(julia_steps) == 2


def test_legacy_ready_state_without_fingerprint_is_reprovisioned(
    runtime_dir, no_gpu, fake_julia, julia_steps
):
    runtime_dir.mkdir(parents=True)
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps(
            {
                "status": "ready",
                "backend": BEAT_CPU,
                "project": str(PACKAGE_DIR / "julia"),
                "julia_executable": str(fake_julia),
            }
        ),
        encoding="utf-8",
    )

    assert provision.provision_cpu(runtime_dir, status_cb=lambda _: None)["status"] == "ready"
    assert len(julia_steps) == 2


def test_a_ready_record_for_a_project_that_moved_is_not_reused(runtime_dir, no_gpu, fake_julia, julia_steps):
    """Reinstalling the package elsewhere leaves the recorded Julia valid.

    The bundled project it instantiated is gone, though, so the record's
    ``ready`` describes a runtime that no longer exists.
    """

    runtime_dir.mkdir(parents=True)
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps(
            {
                "status": "ready",
                "backend": BEAT_CPU,
                "project": "/somewhere/else/hornlab_beat_bem/julia",
                "julia_executable": str(fake_julia),
            }
        ),
        encoding="utf-8",
    )

    assert provision.provision_cpu(runtime_dir, status_cb=lambda _: None)["status"] == "ready"
    assert julia_steps, "a record naming a different project was reused"


def test_reprovisioning_reuses_the_julia_this_directory_holds(runtime_dir, no_gpu, julia_steps, monkeypatch, tmp_path):
    """``--force`` must not re-download the Julia it is about to reuse.

    The in-progress state is written *before* Julia is resolved, so it has
    already overwritten the record ``discover_julia`` would have read -- and
    on the host this path serves there is no Julia on PATH to fall back to.
    """

    unpacked = tmp_path / "runtime-julia" / "bin" / "julia"
    unpacked.parent.mkdir(parents=True)
    unpacked.write_text("", encoding="utf-8")
    runtime_dir.mkdir(parents=True)
    (runtime_dir / provision.STATE_FILENAME).write_text(
        json.dumps(
            {
                "status": "ready",
                "backend": BEAT_CPU,
                "project": str(PACKAGE_DIR / "julia"),
                "julia_executable": str(unpacked),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "discover_julia", lambda explicit=None: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("re-downloaded a Julia the runtime directory already held")

    monkeypatch.setattr(provision, "_download", forbidden)

    state = provision.provision_cpu(runtime_dir, status_cb=lambda _: None, force=True)

    assert state["status"] == "ready"
    assert state["julia_executable"] == str(unpacked)
    assert len(julia_steps) == 2


def test_cpu_failure_is_recorded_and_the_cli_exits_nonzero(runtime_dir, no_gpu, fake_julia, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("CPU probe never solved")

    monkeypatch.setattr(provision, "_run_julia_step", boom)

    assert provision.main(["--backend", "cpu"]) == 1
    stored = provision.read_state(runtime_dir)
    assert stored is not None and stored["status"] == "failed"
    assert "CPU probe never solved" in stored["error"]
    assert provision.provisioned_julia(runtime_dir) is None


def test_cli_backend_cpu_provisions_and_exits_zero(runtime_dir, no_gpu, fake_julia, julia_steps):
    assert provision.main(["--backend", "cpu"]) == 0
    assert provision.read_state(runtime_dir)["backend"] == BEAT_CPU


@pytest.mark.parametrize("gate", ["--if-gpu", "--if-nvidia-gpu"])
def test_cli_refuses_cpu_combined_with_a_gpu_gate(gate, runtime_dir, no_gpu, capsys):
    """Two contradictory instructions get a refusal, not a silent winner."""

    with pytest.raises(SystemExit) as excinfo:
        provision.main(["--backend", "cpu", gate])
    assert excinfo.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err
    assert not runtime_dir.exists()


def test_cli_auto_never_selects_the_cpu_backend(runtime_dir, no_gpu, monkeypatch, capsys):
    """The default is unchanged: no GPU, nothing provisioned, exit 0."""

    def forbidden(*args, **kwargs):
        raise AssertionError("auto provisioned a backend on a GPU-less host")

    monkeypatch.setattr(provision, "provision_cpu", forbidden)
    monkeypatch.setattr(provision, "_ensure_julia", forbidden)
    assert provision.main([]) == 0
    assert "nothing to provision" in capsys.readouterr().out
    assert not runtime_dir.exists()


def test_custom_dir_says_it_will_not_be_discovered(no_gpu, fake_julia, julia_steps, tmp_path, capsys, monkeypatch):
    """Provisioning into ``--dir`` succeeds and then is invisible to discovery."""

    monkeypatch.delenv(provision.RUNTIME_DIR_ENV_VAR, raising=False)
    elsewhere = tmp_path / "elsewhere"
    assert provision.main(["--backend", "cpu", "--dir", str(elsewhere)]) == 0
    out = capsys.readouterr().out
    assert provision.RUNTIME_DIR_ENV_VAR in out and str(elsewhere) in out
    assert provision.read_state(elsewhere)["status"] == "ready"


def test_metal_instantiate_does_not_promise_a_multi_gigabyte_download(runtime_dir, monkeypatch, tmp_path):
    """Metal.jl carries no vendor toolkit, so the CUDA wording would be a lie."""

    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: True)
    fake_julia = tmp_path / "julia"
    fake_julia.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_julia", lambda explicit=None: str(fake_julia))
    labels: list[str] = []
    monkeypatch.setattr(
        provision,
        "_run_julia_step",
        lambda julia, code, **kwargs: labels.append(str(kwargs["label"])),
    )
    provision.provision_gpu(runtime_dir, backend="metal", status_cb=lambda _: None)
    assert labels and not any("several GB" in label for label in labels)


@pytest.mark.parametrize("backend", [BEAT_CPU, BEAT_METAL])
@pytest.mark.parametrize("legacy", [False, True])
def test_upgrade_replaces_old_portable_julia(
    runtime_dir, monkeypatch, julia_steps, backend, legacy
):
    """An old managed unpack is neither discoverable nor ready after upgrade."""
    old = runtime_dir / "julia-1.12.6" / "bin" / "julia"
    old.parent.mkdir(parents=True)
    old.write_text("old runtime", encoding="utf-8")
    project = runtime.default_project(backend)
    previous = {
        "status": "ready", "backend": backend, "project": str(project),
        # Even a matching package fingerprint must not adopt the old runtime.
        "package_fingerprint": runtime.package_fingerprint(project),
        "julia_version": "1.12.6", "julia_executable": str(old),
    }
    if legacy:
        (runtime_dir / provision.STATE_FILENAME).write_text(json.dumps(previous))
    else:
        provision._write_state(runtime_dir, previous)
    monkeypatch.delenv(runtime.JULIA_ENV_VAR, raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)
    monkeypatch.setattr(provision, "_gpu_hardware_present", lambda _: True)
    assert provision.provisioned_julia(runtime_dir) is None
    assert runtime.discover_julia() is None
    assert not provision.backend_ready(backend, runtime_dir)
    downloaded = []
    monkeypatch.setattr(provision, "_official_checksum", lambda *args: "a" * 64)
    monkeypatch.setattr(provision, "_download", lambda *args, **kwargs: downloaded.append(kwargs["expected_sha256"]))
    current = runtime_dir / f"julia-{provision.JULIA_VERSION}" / "bin" / "julia"
    def extract(*args):
        current.parent.mkdir(parents=True)
        current.write_text("new runtime", encoding="utf-8")
        return current
    monkeypatch.setattr(provision, "_extract_julia", extract)
    state = (provision.provision_cpu(runtime_dir, status_cb=lambda _: None)
             if backend == BEAT_CPU else
             provision.provision_gpu(runtime_dir, backend=backend, status_cb=lambda _: None))
    assert state["status"] == "ready"
    assert state["julia_executable"] == str(current)
    assert state["julia_version"] == provision.JULIA_VERSION
    assert len(downloaded) == 1
    assert len(julia_steps) == 2
    assert runtime.discover_julia() == str(current)
    assert provision.backend_ready(backend, runtime_dir)
    assert old.exists()  # existing workers may still use it


def test_current_portable_julia_is_reused_after_failed_provision(
    runtime_dir, no_gpu, monkeypatch, julia_steps
):
    current = runtime_dir / f"julia-{provision.JULIA_VERSION}" / "bin" / "julia"
    current.parent.mkdir(parents=True)
    current.write_text("", encoding="utf-8")
    provision._write_state(runtime_dir, {
        "status": "failed", "backend": BEAT_CPU, "julia_executable": str(current),
    })
    monkeypatch.delenv(runtime.JULIA_ENV_VAR, raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("downloaded the current managed runtime again")
    monkeypatch.setattr(provision, "_download", forbidden)
    state = provision.provision_cpu(runtime_dir, status_cb=lambda _: None)
    assert state["status"] == "ready"
    assert state["julia_executable"] == str(current)


def test_explicit_old_julia_still_wins(runtime_dir, monkeypatch):
    old = runtime_dir / "julia-1.12.6" / "bin" / "julia"
    old.parent.mkdir(parents=True)
    old.write_text("", encoding="utf-8")
    monkeypatch.setenv(runtime.JULIA_ENV_VAR, str(old))
    assert provision._ensure_julia(runtime_dir, lambda _: None) == str(old)


def test_portable_downloads_pin_the_current_release():
    assert provision.JULIA_VERSION == "1.12.7"
    windows = provision._JULIA_DOWNLOADS[("Windows", "AMD64")]
    assert windows["filename"] == "julia-1.12.7-win64.zip"
    assert windows["sha256"] == "ff5c7eb354c2fcb48401114a5fbcfe8e60181f95d9af42b266f975265a5bad47"
    for entry in provision._JULIA_DOWNLOADS.values():
        assert provision.JULIA_VERSION in entry["url"]
        assert provision.JULIA_VERSION in entry["filename"]


def test_unmanaged_install_inside_runtime_dir_is_preserved(runtime_dir):
    external = runtime_dir / "julia-custom" / "bin" / "julia"
    external.parent.mkdir(parents=True)
    external.write_text("", encoding="utf-8")
    assert provision._recorded_julia({"julia_executable": str(external)}, runtime_dir) == str(external)
