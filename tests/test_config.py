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
    # Metal has no near-pair kernel either, and unlike ROCm it used to accept
    # the flag: BeatEngineCore's :metal branch forwards no near-correction
    # cache and the Metal assembly takes none, so the solve announced a
    # correction it never applied. Measured 2026-09-05 on a 320-face sphere at
    # 2 kHz: flag on against flag off agreed to 1.8e-7 relative -- that
    # backend's own atomics noise -- while the CPU backend moved by 1.4e-4.
    with pytest.raises(NotImplementedError, match="Metal"):
        SolveConfig(near_correction=True, beat_backend="metal")
    # Off, it is not a refusal on any backend: the flag is what is refused.
    for backend in ("cpu", "cuda", "rocm", "metal"):
        SolveConfig(near_correction=False, beat_backend=backend)


def test_quadrature_order_is_restricted_to_the_orders_the_engine_has():
    """`triangle_rule` has three rules; every other order is the middle one.

    Orders 1, 2 and 4 select the 1-, 3- and 6-point rules. Everything else
    falls through to the 3-point rule, so `quadrature_order=6` was accepted,
    passed to the solver, and solved *less* accurately than the default while
    reading like a refinement.
    """

    from hornlab_beat_bem.config import SUPPORTED_QUADRATURE_ORDERS

    assert SUPPORTED_QUADRATURE_ORDERS == (1, 2, 4)
    for order in SUPPORTED_QUADRATURE_ORDERS:
        assert _request_payload_config(SolveConfig(quadrature_order=order))[
            "quadrature_order"
        ] == order
    for order in (0, -1, 3, 5, 6, 8, 2.5):
        with pytest.raises(ValueError, match="quadrature_order"):
            SolveConfig(quadrature_order=order)


def test_a_complex_source_amplitude_is_the_documented_refusal_not_a_typeerror():
    """The report declares complex drives refused, so refuse them as one.

    `float(amplitude)` turned a complex amplitude into a bare TypeError while
    every other refusal here raises NotImplementedError, so a caller catching
    the documented type caught nothing.
    """

    with pytest.raises(NotImplementedError, match="complex"):
        SolveConfig(velocity_sources={2: 1.0 + 0.5j})
    with pytest.raises(NotImplementedError, match="complex"):
        SolveConfig(velocity_sources={2: complex(1.0, 0.0)})
    with pytest.raises(ValueError, match="real number"):
        SolveConfig(velocity_sources={2: "1.0j"})
    assert SolveConfig(velocity_sources={2: 1}).source_tag == 2


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

def _axis_aligned_frame(origin):
    import numpy as np

    from hornlab_beat_bem import ObservationFrame

    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.asarray(origin, dtype=float),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
    )


def _payload_for_frame(config: SolveConfig) -> dict:
    """The ``config`` block including the translation the frame asks for."""

    import numpy as np

    from hornlab_beat_bem.sweep import _validated_frame_translation

    return _request_payload(
        "mesh.msh",
        np.asarray([1000.0]),
        config,
        translation=_validated_frame_translation(config.frame_override),
    )["config"]


def test_observation_origin_defaults_to_the_frame_or_the_mesh_origin():
    """No named origin means no claim: the arcs sit where the frame puts them."""

    assert ObservationConfig().origin is None
    payload = _request_payload_config(SolveConfig())
    assert payload["meshes"][0]["translation_m"] == [0.0, 0.0, 0.0]


def test_named_observation_origin_without_a_frame_is_refused():
    """It used to be accepted and then ignored -- identical requests either way."""

    for named in ("mouth", "throat"):
        with pytest.raises(NotImplementedError, match="needs an explicit frame_override"):
            SolveConfig(observation=ObservationConfig(origin=named))


def test_named_observation_origin_is_realised_by_the_frame():
    """An axially displaced horn: mouth and throat now move the mesh apart."""

    mouth = SolveConfig(
        observation=ObservationConfig(origin="mouth"),
        frame_override=_axis_aligned_frame([0.0, 0.0, 0.32]),
    )
    throat = SolveConfig(
        observation=ObservationConfig(origin="throat"),
        frame_override=_axis_aligned_frame([0.0, 0.0, 0.0]),
    )
    assert _payload_for_frame(mouth)["meshes"][0]["translation_m"] == [0.0, 0.0, -0.32]
    assert _payload_for_frame(throat)["meshes"][0]["translation_m"] == [0.0, 0.0, 0.0]
    assert _payload_for_frame(mouth) != _payload_for_frame(throat)


def test_observation_origin_rejects_an_unknown_feature():
    with pytest.raises(ValueError, match="origin must be"):
        ObservationConfig(origin="baffle")


def test_the_origin_refusal_survives_mutation_after_construction():
    """``solve_frequencies`` re-checks, so dropping the frame later still fails."""

    from hornlab_beat_bem.config import reject_unrepresentable_observation_origin

    config = SolveConfig(
        observation=ObservationConfig(origin="throat"),
        frame_override=_axis_aligned_frame([0.0, 0.0, 0.0]),
    )
    reject_unrepresentable_observation_origin(config)
    config.frame_override = None
    with pytest.raises(NotImplementedError, match="needs an explicit frame_override"):
        reject_unrepresentable_observation_origin(config)
