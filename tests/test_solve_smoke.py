"""End-to-end CPU solve through the real Julia runtime (slow)."""

from pathlib import Path
import tempfile

import numpy as np
import pytest

import hornlab_beat_bem as beat
from hornlab_beat_bem.sweep import _WARMUP_TETRAHEDRON


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def julia():
    executable = beat.discover_julia()
    if executable is None:
        pytest.skip("no Julia executable (set HORNLAB_BEAT_JULIA)")
    return executable


def test_cpu_solve_tetrahedron(julia):
    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_path = Path(temp_dir) / "warmup.msh"
        mesh_path.write_text(_WARMUP_TETRAHEDRON, encoding="utf-8")
        streamed = []
        config = beat.SolveConfig(
            beat_backend="cpu",
            julia_executable=julia,
            observation=beat.ObservationConfig(
                planes=["horizontal", "vertical"],
                distance_m=1.0,
                angle_min_deg=0.0,
                angle_max_deg=90.0,
                angle_count=4,
            ),
            on_frequency_result=lambda index, freq, entry: bool(
                streamed.append((index, freq)) or True
            ),
        )
        result = beat.solve_frequencies(mesh_path, [300.0, 500.0], config)
    try:
        assert result.frequencies_hz.tolist() == [300.0, 500.0]
        assert result.pressure_complex.shape == (2, 2, 4)
        assert result.spl_db.shape == (2, 2, 4)
        assert np.all(np.isfinite(result.pressure_complex))
        assert np.all(np.isfinite(result.impedance))
        # A pulsating source's near-symmetric tetrahedron: both cuts agree on
        # axis, and the throat sees positive radiation resistance.
        assert np.allclose(
            np.abs(result.pressure_complex[:, 0, 0]),
            np.abs(result.pressure_complex[:, 1, 0]),
            rtol=1e-3,
        )
        omega = 2.0 * np.pi * result.frequencies_hz
        z_specific = np.conjugate(-1j * omega * result.impedance)
        assert np.all(z_specific.real > 0.0)
        assert streamed == [(0, 300.0), (1, 500.0)]
        assert len(result.solver_log) == 2
    finally:
        beat.shutdown_workers()
