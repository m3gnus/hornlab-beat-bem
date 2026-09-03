import pytest

from hornlab_beat_bem import (
    ObservationConfig,
    SolveConfig,
    beat_symmetry_mode,
    reject_unsupported_native_symmetry,
)
from hornlab_beat_bem.sweep import _request_payload


def _request_payload_config(config: SolveConfig) -> dict:
    """The ``config`` block the Julia solver actually receives."""

    import numpy as np

    return _request_payload(
        "mesh.msh", np.asarray([1000.0]), config, translation=(0.0, 0.0, 0.0)
    )["config"]


def test_observation_defaults_are_valid():
    config = ObservationConfig()
    assert config.planes == ["horizontal", "vertical"]
    assert config.step_deg == pytest.approx(5.0)


def test_observation_accepts_diagonal_plane_with_inclination():
    config = ObservationConfig(
        planes=["horizontal", "vertical", "diagonal"], inclination_deg=30.0
    )
    assert config.inclination_deg == 30.0


def test_observation_rejects_unknown_plane():
    with pytest.raises(ValueError, match="unknown"):
        ObservationConfig(planes=["horizontal", "sideways"])


def test_observation_sphere_grid_validation():
    config = ObservationConfig(sphere_grid=(37, 72))
    assert config.sphere_grid == (37, 72)
    with pytest.raises(ValueError, match="n_theta"):
        ObservationConfig(sphere_grid=(1, 72))
    with pytest.raises(ValueError, match="n_phi"):
        ObservationConfig(sphere_grid=(37, 2))
    with pytest.raises(ValueError, match="sphere_theta_max_deg"):
        ObservationConfig(sphere_grid=(37, 72), sphere_theta_max_deg=0.0)


def test_observation_requires_zero_degree_sample():
    with pytest.raises(ValueError, match="0 degrees"):
        ObservationConfig(angle_min_deg=10.0, angle_max_deg=90.0)


def test_observation_rejects_out_of_range_angles():
    with pytest.raises(ValueError):
        ObservationConfig(angle_min_deg=-200.0, angle_max_deg=180.0)


@pytest.mark.parametrize(
    ("plane", "mode"),
    [(None, "off"), ("yz", "x"), ("yz+xz", "xy")],
)
def test_symmetry_mapping(plane, mode):
    assert beat_symmetry_mode(plane) == mode


@pytest.mark.parametrize("plane", ["xz", "xy"])
def test_unrepresentable_symmetry_is_rejected(plane):
    config = SolveConfig(native_symmetry_plane=plane)
    with pytest.raises(NotImplementedError):
        reject_unsupported_native_symmetry(config)


def test_solve_config_accepts_axial_and_rejects_unknown_motion():
    assert SolveConfig(source_motion="axial").source_motion == "axial"
    with pytest.raises(ValueError):
        SolveConfig(source_motion="sideways")


def test_solve_config_rejects_non_unit_amplitude():
    with pytest.raises(NotImplementedError):
        SolveConfig(velocity_sources={2: 0.5})


def test_solve_config_rejects_multiple_sources():
    with pytest.raises(NotImplementedError):
        SolveConfig(velocity_sources={2: 1.0, 3: 1.0})


def test_source_tag_reflects_velocity_sources():
    assert SolveConfig(velocity_sources={7: 1.0}).source_tag == 7


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm", "metal"])
def test_solve_config_accepts_every_declared_backend(backend):
    assert SolveConfig(beat_backend=backend).beat_backend == backend


def test_solve_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="beat_backend"):
        SolveConfig(beat_backend="opencl")


def test_every_declared_backend_has_a_bundled_julia_project():
    from hornlab_beat_bem import BEAT_BACKENDS
    from hornlab_beat_bem.runtime import default_project

    for backend in BEAT_BACKENDS:
        project = default_project(backend)
        assert (project / "Project.toml").exists(), backend


def test_near_correction_defaults_off_and_stays_out_of_the_request():
    config = SolveConfig()
    assert config.near_correction is False
    assert _request_payload_config(config).keys().isdisjoint(
        {"near_correction_enabled", "near_correction_cutoff", "near_correction_order"}
    )


def test_near_correction_reaches_the_solver_request():
    config = SolveConfig(
        near_correction=True, near_correction_cutoff=1.5, near_correction_order=10
    )
    solver_config = _request_payload_config(config)
    assert solver_config["near_correction_enabled"] is True
    assert solver_config["near_correction_cutoff"] == pytest.approx(1.5)
    assert solver_config["near_correction_order"] == 10


def test_near_correction_validation():
    with pytest.raises(ValueError, match="near_correction_cutoff"):
        SolveConfig(near_correction=True, near_correction_cutoff=0.0)
    with pytest.raises(ValueError, match="near_correction_order"):
        SolveConfig(near_correction=True, near_correction_order=3)
    # The vendored ROCm assembly has no near-pair kernel, so accepting the flag
    # there would report a corrected solve that never ran the correction.
    with pytest.raises(NotImplementedError, match="ROCm"):
        SolveConfig(near_correction=True, beat_backend="rocm")
    # CUDA has the kernel but still takes a single image-near cache upstream,
    # so under `yz+xz` it would leave two of three mirror transforms
    # uncorrected and say nothing. The multi-cache patch is applied to the CPU
    # assembly only, because no machine here has an NVIDIA GPU to verify a
    # device-side edit on.
    with pytest.raises(NotImplementedError, match="CUDA"):
        SolveConfig(near_correction=True, beat_backend="cuda")


def test_solve_precision_is_cpu_only():
    assert SolveConfig().solve_precision == "single"
    assert "solve_precision" not in _request_payload_config(SolveConfig())
    config = SolveConfig(solve_precision="double")
    assert _request_payload_config(config)["solve_precision"] == "double"
    with pytest.raises(NotImplementedError, match="CPU backend"):
        SolveConfig(solve_precision="double", beat_backend="cuda")
    with pytest.raises(ValueError, match="solve_precision"):
        SolveConfig(solve_precision="extended")


def test_singular_order_above_four_requires_double_precision():
    # Measured on the ASRO quarter mesh: order 4 is converged to 0.0016 dB rms
    # in Float64, while in Float32 order 8 is 0.031 dB *worse* than its own
    # double-precision answer. Allowing it on the GPU path would be a
    # pessimisation dressed up as an accuracy knob.
    with pytest.raises(ValueError, match="solve_precision='double'"):
        SolveConfig(singular_order=8)
    assert SolveConfig(singular_order=8, solve_precision="double").singular_order == 8
    with pytest.raises(ValueError, match="singular_order"):
        SolveConfig(singular_order=13, solve_precision="double")