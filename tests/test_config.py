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


def test_observation_rejects_diagonal_plane():
    with pytest.raises(ValueError, match="diagonal"):
        ObservationConfig(planes=["horizontal", "diagonal"])


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


def test_solve_config_rejects_axial_source_motion():
    with pytest.raises(NotImplementedError):
        SolveConfig(source_motion="axial")


def test_solve_config_rejects_non_unit_amplitude():
    with pytest.raises(NotImplementedError):
        SolveConfig(velocity_sources={2: 0.5})


def test_solve_config_rejects_multiple_sources():
    with pytest.raises(NotImplementedError):
        SolveConfig(velocity_sources={2: 1.0, 3: 1.0})


def test_source_tag_reflects_velocity_sources():
    assert SolveConfig(velocity_sources={7: 1.0}).source_tag == 7
