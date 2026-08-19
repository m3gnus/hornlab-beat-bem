"""Pure-math checks of the drive-convention and frame mapping."""

import numpy as np
import pytest

from hornlab_beat_bem.config import ObservationFrame
from hornlab_beat_bem.sweep import (
    _BEAT_IMPEDANCE_FORCE_FACTOR,
    _validated_frame_translation,
)


def test_frame_translation_moves_origin_to_solver_origin():
    frame = ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.array([0.0, 0.0, 0.25]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
    )
    assert _validated_frame_translation(frame) == (0.0, 0.0, -0.25)


def test_frame_translation_none_is_identity():
    assert _validated_frame_translation(None) == (0.0, 0.0, 0.0)


def test_tilted_frame_is_rejected():
    frame = ObservationFrame(
        axis=np.array([0.1, 0.0, 0.99498743710662]),
        origin=np.zeros(3),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
    )
    with pytest.raises(NotImplementedError, match="tilted"):
        _validated_frame_translation(frame)


def test_impedance_wire_reconstruction_round_trip():
    """The [Re(F)/2, -Im(F)/2] packing must invert exactly.

    Simulates the Julia side: mean surface pressure p under a unit velocity
    drive on a reduced-domain source of area A with symmetry factor s produces
    the wire pair; the sweep-side reconstruction must return p.
    """

    p_mean = 3.5 - 1.25j
    area_reduced = 0.004
    sym_factor = 4
    force = _BEAT_IMPEDANCE_FORCE_FACTOR * sym_factor * (p_mean * area_reduced)
    wire = (force.real / 2.0, -force.imag / 2.0)

    reconstructed_force = 2.0 * wire[0] - 2.0j * wire[1]
    reconstructed = reconstructed_force / (
        _BEAT_IMPEDANCE_FORCE_FACTOR * sym_factor * area_reduced
    )
    assert reconstructed == pytest.approx(p_mean)


def test_acceleration_scaling_matches_wg_impedance_contract():
    """(i/omega) rescale + WG's conj(-i*omega*z)/rho_c == conj(p_vel)/rho_c."""

    rng = np.random.default_rng(7)
    p_vel = rng.normal(size=8) + 1j * rng.normal(size=8)
    omega = 2.0 * np.pi * np.geomspace(100.0, 10_000.0, 8)
    rho_c = 1.2041 * 343.0

    p_accel = p_vel / (-1j * omega)
    wg_mapped = np.conjugate(-1j * omega * p_accel) / rho_c
    assert np.allclose(wg_mapped, np.conjugate(p_vel) / rho_c)
