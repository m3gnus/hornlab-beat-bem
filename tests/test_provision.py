"""GPU-gated provisioning: no hardware means no download, ever."""

from __future__ import annotations

import json

import pytest

from hornlab_beat_bem import provision, runtime


@pytest.fixture()
def runtime_dir(tmp_path, monkeypatch):
    directory = tmp_path / "runtime"
    monkeypatch.setenv(provision.RUNTIME_DIR_ENV_VAR, str(directory))
    return directory


def test_no_gpu_means_no_download_and_no_state(runtime_dir, monkeypatch):
    monkeypatch.setattr(runtime, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setattr(runtime, "_apple_gpu_present", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("provisioning attempted a download without a GPU")

    monkeypatch.setattr(provision, "_ensure_julia", forbidden)
    monkeypatch.setattr(provision, "_download", forbidden)

    state = provision.provision_cuda(runtime_dir, status_cb=lambda _: None)
    assert state["status"] == "skipped"
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
            {"status": "ready", "backend": "cuda", "julia_executable": str(fake_julia)}
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
