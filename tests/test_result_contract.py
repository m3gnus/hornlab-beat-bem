"""The result contract: absolute SPL versus normalized directivity.

``directivity_db`` is hornlab-metal-bem's name for the on-axis-normalized
pattern, and a consumer written against that package reads this one through
the same name. It used to return ``spl_db`` unchanged here, which is absolute
SPL -- roughly a 94 dB offset for a 1 Pa reference, in a field whose whole
point is that 0 dB means "as loud as on axis".
"""

import numpy as np
import pytest

from hornlab_beat_bem import MeshInfo, SolveConfig, SolveResult
from hornlab_beat_bem._constants import REFERENCE_PRESSURE


def _result(pressure, angles, *, planes=None) -> SolveResult:
    pressure = np.asarray(pressure, dtype=np.complex128)
    angles = np.asarray(angles, dtype=np.float64)
    with np.errstate(divide="ignore"):
        spl_db = 20.0 * np.log10(np.abs(pressure) / REFERENCE_PRESSURE)
    return SolveResult(
        frequencies_hz=np.arange(pressure.shape[0], dtype=np.float64) + 1000.0,
        pressure_complex=pressure,
        spl_db=spl_db,
        impedance=np.ones(pressure.shape[0], dtype=np.complex128),
        observation_angles_deg=angles,
        observation_planes=planes or ["horizontal"],
        config=SolveConfig(),
        mesh_info=MeshInfo(0, 0, {}),
    )


def test_spl_db_stays_absolute_and_directivity_is_normalized():
    """The review probe's example, with both quantities kept apart."""

    result = _result([[[1.0 + 0j, 0.5 + 0j]]], [0.0, 90.0])
    assert result.spl_db[0, 0].tolist() == pytest.approx([93.9794000867, 87.9588001734])
    assert result.directivity_db[0, 0].tolist() == pytest.approx([0.0, -6.0205999132])
    assert result.spl_norm_db is not None
    assert np.array_equal(result.spl_norm_db, result.directivity_db)


def test_directivity_is_invariant_to_a_nonzero_reference_level():
    """Scaling the whole field moves SPL and leaves the pattern alone."""

    quiet = _result([[[1.0 + 0j, 0.5 + 0j, 0.25 + 0j]]], [0.0, 45.0, 90.0])
    loud = _result([[[733.0 + 0j, 366.5 + 0j, 183.25 + 0j]]], [0.0, 45.0, 90.0])
    assert np.max(np.abs(loud.spl_db - quiet.spl_db)) > 50.0
    assert loud.directivity_db == pytest.approx(quiet.directivity_db)
    assert loud.directivity_db[0, 0].tolist() == pytest.approx(
        [0.0, -6.0205999132, -12.0411998265]
    )


def test_reference_is_the_on_axis_sample_under_negative_angle_ordering():
    """A cut that starts at -90 normalizes on 0, not on its first sample."""

    pressure = [[[0.25 + 0j, 0.5 + 0j, 1.0 + 0j, 0.5 + 0j, 0.25 + 0j]]]
    result = _result(pressure, [-90.0, -45.0, 0.0, 45.0, 90.0])
    assert result.directivity_reference_index == 2
    assert result.directivity_reference_deg == 0.0
    assert result.directivity_db[0, 0].tolist() == pytest.approx(
        [-12.0411998265, -6.0205999132, 0.0, -6.0205999132, -12.0411998265]
    )


def test_reference_of_a_grid_that_straddles_zero_is_the_first_closest_sample():
    """No sample sits on axis; the tie resolves left, and it is reported."""

    result = _result([[[1.0 + 0j, 0.5 + 0j]]], [-10.0, 10.0])
    assert result.directivity_reference_index == 0
    assert result.directivity_reference_deg == -10.0
    assert result.directivity_db[0, 0].tolist() == pytest.approx([0.0, -6.0205999132])


def test_a_silent_reference_is_floored_rather_than_infinite():
    """A null on axis must not turn the whole cut into ``nan``."""

    result = _result([[[0.0 + 0j, 1.0 + 0j]]], [0.0, 90.0])
    assert np.isneginf(result.spl_db[0, 0, 0])
    directivity = result.directivity_db[0, 0]
    assert np.all(np.isfinite(directivity))
    assert directivity[0] == 0.0
    # The floor is -120 dB SPL, so a 1 Pa sample stands that far above it
    # plus its own 93.98 dB -- implausibly loud, and legible as such.
    assert directivity[1] == pytest.approx(213.9794000867)


def test_directivity_normalizes_each_frequency_and_plane_separately():
    pressure = np.array(
        [
            [[1.0 + 0j, 0.5 + 0j], [2.0 + 0j, 2.0 + 0j]],
            [[4.0 + 0j, 1.0 + 0j], [0.5 + 0j, 0.25 + 0j]],
        ],
        dtype=np.complex128,
    )
    result = _result(pressure, [0.0, 90.0], planes=["horizontal", "vertical"])
    assert result.directivity_db[:, :, 0].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert result.directivity_db[:, :, 1] == pytest.approx(
        np.array([[-6.0205999132, 0.0], [-12.0411998265, -6.0205999132]])
    )


def test_directivity_matches_the_hornlab_metal_bem_formula():
    """Same reference, same floor, same subtraction -- not merely similar."""

    rng = np.random.default_rng(20260904)
    pressure = rng.normal(size=(3, 2, 7)) + 1j * rng.normal(size=(3, 2, 7))
    angles = np.linspace(-90.0, 90.0, 7)
    result = _result(pressure, angles)
    result.observation_planes = ["horizontal", "vertical"]

    floor = REFERENCE_PRESSURE * 10.0 ** (-120.0 / 20.0)
    spl_raw = 20.0 * np.log10(np.maximum(np.abs(pressure), floor) / REFERENCE_PRESSURE)
    on_axis = int(np.argmin(np.abs(angles)))
    expected = spl_raw - spl_raw[:, :, on_axis][:, :, None]
    assert result.directivity_db == pytest.approx(expected)


def test_directivity_refuses_a_pressure_block_that_does_not_match_its_angles():
    result = _result([[[1.0 + 0j, 0.5 + 0j]]], [0.0, 45.0, 90.0])
    with pytest.raises(ValueError, match="angle axis"):
        _ = result.directivity_db


def test_partial_flags_default_to_a_complete_result():
    result = _result([[[1.0 + 0j, 0.5 + 0j]]], [0.0, 90.0])
    assert result.cancelled is False
    assert result.requested_frequency_count is None
    assert result.is_partial is False
