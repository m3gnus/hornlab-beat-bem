import pytest

from hornlab_beat_bem import (
    ObservationConfig,
    SolveConfig,
    beat_symmetry_mode,
    reject_unsupported_native_symmetry,
)


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
