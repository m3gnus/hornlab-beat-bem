"""End-to-end CPU solve through the real Julia runtime (slow)."""

import tempfile
from pathlib import Path

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


def test_cpu_regular_kernel_switch_reaches_the_worker(julia, monkeypatch):
    """The vectorised regular kernel is the default, and the switch is real.

    The tetrahedron above cannot show this: all four of its faces touch, so it
    has no regular pair at all. The bundled sample mesh has them. The two
    kernels differ in summation order and in the sincos they call, so they
    agree to Float32 rounding and not bitwise -- and if they were bitwise
    equal here, the switch would not be reaching the engine.
    """

    mesh_path = (
        Path(beat.__file__).resolve().parent / "julia" / "test_meshes" / "sample.msh"
    )

    def solve(kernel):
        if kernel is None:
            monkeypatch.delenv("BLAB_BEAT_CPU_REGULAR_KERNEL", raising=False)
        else:
            monkeypatch.setenv("BLAB_BEAT_CPU_REGULAR_KERNEL", kernel)
        config = beat.SolveConfig(
            beat_backend="cpu",
            julia_executable=julia,
            mesh_scale=0.001,
            velocity_sources={2: 1.0},
            observation=beat.ObservationConfig(
                planes=["horizontal", "vertical"],
                distance_m=1.0,
                angle_min_deg=0.0,
                angle_max_deg=90.0,
                angle_count=4,
            ),
        )
        try:
            return beat.solve_frequencies(mesh_path, [800.0, 6000.0], config)
        finally:
            beat.shutdown_workers()

    default = solve(None)
    scalar = solve("scalar")

    def kernels(result):
        return [entry["native_diagnostics"]["cpu_regular_kernel"] for entry in result.solver_log]

    assert kernels(default) == ["simd", "simd"]
    assert kernels(scalar) == ["scalar", "scalar"]
    assert np.all(np.isfinite(default.pressure_complex))
    difference = np.linalg.norm(default.pressure_complex - scalar.pressure_complex)
    reference = np.linalg.norm(scalar.pressure_complex)
    assert 0.0 < difference <= 1e-4 * reference
