"""The rigid half-space seam: what it accepts, what it refuses, and why.

BEAT's ground plane is a SEPARATE configuration axis from native symmetry,
with separate machinery, because they are different physics that share the
solver's one ``symmetry`` field. Conflating them is how the 6.02 dB impedance
defect got in, so most of what is asserted here is that they stay apart.

The numerical half of the ground plane is gated by
``julia/scripts/validate_analytic_exterior.jl`` against a closed form, not
here.
"""
from __future__ import annotations

import numpy as np
import pytest

from hornlab_beat_bem import (
    GROUND_PLANE_AXES,
    GroundPlane,
    ObservationFrame,
    SolveConfig,
    beat_image_mode,
    ground_plane_enabled,
    reject_unsupported_ground_plane,
)
from hornlab_beat_bem.sweep import _SYMMETRY_FACTOR, _ground_placed_translation, _request_payload


def _floor(**overrides) -> SolveConfig:
    payload = {"enabled": True, "axis": "y", "height_m": 1.0}
    payload.update(overrides)
    return SolveConfig(ground_plane=payload)


def _config_block(config: SolveConfig, translation=(0.0, 0.0, 0.0)) -> dict:
    return _request_payload(
        "mesh.msh", np.asarray([1000.0]), config, translation=translation
    )["config"]


# --- the wire shape WG sends -------------------------------------------------


def test_a_dict_is_coerced_to_a_ground_plane():
    """WG sends ``ground_plane: {enabled, axis, height_m}`` as a mapping."""
    config = _floor()
    assert isinstance(config.ground_plane, GroundPlane)
    assert (config.ground_plane.axis, config.ground_plane.height_m) == ("y", 1.0)


def test_enabled_is_the_on_off_signal_not_the_presence_of_an_axis():
    """A disabled ground plane still carries axis 'y'; it must solve free field."""
    config = SolveConfig(ground_plane={"enabled": False, "axis": "y", "height_m": 1.0})
    assert config.ground_plane is not None
    assert not ground_plane_enabled(config)
    assert beat_image_mode(config) == "off"
    assert _config_block(config)["symmetry"] == "off"
    # ...and a disabled plane cannot refuse anything, whatever axis it names.
    reject_unsupported_ground_plane(
        SolveConfig(ground_plane={"enabled": False, "axis": "x"})
    )


def test_an_enabled_floor_reaches_the_solver_as_the_ground_mode():
    config = _floor()
    assert ground_plane_enabled(config)
    assert beat_image_mode(config) == "ground"
    block = _config_block(config)
    assert block["symmetry"] == "ground"
    assert block["ground_plane_min_clearance_m"] == pytest.approx(0.0)


def test_min_clearance_reaches_the_julia_domain_guard():
    block = _config_block(_floor(min_clearance_m=0.02))
    assert block["ground_plane_min_clearance_m"] == pytest.approx(0.02)


def test_a_free_field_solve_carries_no_ground_key_at_all():
    assert "ground_plane_min_clearance_m" not in _config_block(SolveConfig())


# --- the refusals ------------------------------------------------------------


@pytest.mark.parametrize("axis", ["x", "z"])
def test_the_unsupported_axes_are_refused_by_name_not_substituted(axis):
    """BEAT mirrors across Y=0 only. Silently using the floor for a wall would
    answer confidently about the wrong room, and bempp-bem does all three, so
    this refusal fires in production."""
    with pytest.raises(NotImplementedError) as excinfo:
        reject_unsupported_ground_plane(_floor(axis=axis))
    message = str(excinfo.value)
    assert f"axis={axis!r}" in message
    assert "y = 0 only" in message
    assert "BEMPP" in message


def test_the_capability_advertisement_names_axes_rather_than_a_boolean():
    assert GROUND_PLANE_AXES == ("y",)
    reject_unsupported_ground_plane(_floor(axis="y"))
    for axis in set("xyz") - set(GROUND_PLANE_AXES):
        with pytest.raises(NotImplementedError):
            reject_unsupported_ground_plane(_floor(axis=axis))


@pytest.mark.parametrize("plane", ["yz", "yz+xz"])
def test_a_ground_plane_does_not_compose_with_native_symmetry(plane):
    """A y ground plane lifts the model clear of y=0, destroying the xz mirror;
    and the solver carries one image-transform set, so even the surviving yz
    half cannot ride alongside the ground image."""
    config = SolveConfig(
        ground_plane={"enabled": True, "axis": "y", "height_m": 1.0},
        native_symmetry_plane=plane,
    )
    with pytest.raises(NotImplementedError) as excinfo:
        reject_unsupported_ground_plane(config)
    assert "single image-transform set" in str(excinfo.value)


def test_native_symmetry_alone_is_untouched_by_the_ground_seam():
    config = SolveConfig(native_symmetry_plane="yz+xz")
    reject_unsupported_ground_plane(config)
    assert beat_image_mode(config) == "xy"


def test_the_legacy_hornlab_xy_refusal_is_still_in_place():
    """'xy' means three different things across this workspace; none of the
    ground work may soften the existing refusal of HornLab's."""
    from hornlab_beat_bem import reject_unsupported_native_symmetry

    for plane in ("xy", "xz"):
        with pytest.raises(NotImplementedError):
            reject_unsupported_native_symmetry(SolveConfig(native_symmetry_plane=plane))


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"axis": "w"},
        {"height_m": float("nan")},
        {"height_m": float("inf")},
        {"min_clearance_m": -1.0},
        {"enabled": "yes"},
    ],
)
def test_invalid_ground_planes_are_rejected(payload):
    with pytest.raises(ValueError):
        _floor(**payload)


def test_ground_plane_must_be_a_mapping_or_a_ground_plane():
    with pytest.raises(ValueError):
        SolveConfig(ground_plane="y")


# --- placement ---------------------------------------------------------------


def test_height_lifts_the_model_along_the_ground_axis():
    """The plane passes through the solver origin and the MODEL is translated,
    so the containment check downstream is min(coord) + height_m >= 0."""
    assert _ground_placed_translation((0.0, 0.0, 0.0), _floor(height_m=1.25)) == (
        0.0,
        1.25,
        0.0,
    )


def test_a_frame_that_also_moves_along_the_ground_axis_is_refused():
    """Both would place the mesh along Y and the later one would win in
    silence, landing the body at a height nobody asked for."""
    frame = ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.array([0.0, 0.4, 0.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
    )
    config = SolveConfig(
        ground_plane={"enabled": True, "axis": "y", "height_m": 1.0},
        frame_override=frame,
    )
    with pytest.raises(NotImplementedError) as excinfo:
        _ground_placed_translation((0.0, -0.4, 0.0), config)
    assert "must be zero" in str(excinfo.value)


def test_a_frame_on_the_horn_axis_composes_with_the_ground_plane():
    """WG's ordinary case: z is the horn axis, so the observation origin's y
    component is already zero and only x and z come from the frame."""
    assert _ground_placed_translation((0.1, 0.0, -0.06), _floor(height_m=0.9)) == (
        0.1,
        0.9,
        -0.06,
    )


def test_no_ground_plane_leaves_the_frame_translation_alone():
    assert _ground_placed_translation((0.1, 0.2, 0.3), SolveConfig()) == (0.1, 0.2, 0.3)


# --- the impedance defect ----------------------------------------------------


def test_the_ground_image_counts_as_one_radiator_not_two():
    """The Python half of the 6.02 dB fix.

    ``impedance_for_radiators`` in ``julia/BeatEngineDriver.jl`` no longer
    scales the integrated force by ``symmetry_reduction_factor(:ground)``, and
    ``_SYMMETRY_FACTOR`` here divides by the matching 1. The two constants are
    one fix in two files: a 2 in one and a 1 in the other is a 6.02 dB error in
    reported impedance over a completely correct pressure field.
    """
    assert _SYMMETRY_FACTOR["ground"] == 1
    # The mirror modes are unchanged: their images ARE real radiators.
    assert _SYMMETRY_FACTOR["off"] == 1
    assert _SYMMETRY_FACTOR["x"] == 2
    assert _SYMMETRY_FACTOR["xy"] == 4


def test_every_reachable_image_mode_has_a_symmetry_factor():
    """A mode the solver accepts but this table does not would raise KeyError
    mid-sweep, after the solve."""
    modes = {
        beat_image_mode(SolveConfig(native_symmetry_plane=plane))
        for plane in (None, "yz", "yz+xz")
    } | {beat_image_mode(_floor())}
    assert modes <= set(_SYMMETRY_FACTOR)
    assert modes == {"off", "x", "xy", "ground"}


# --- end to end through the real solver --------------------------------------

_GROUND_FREQUENCIES = [300.0, 500.0]


@pytest.fixture(scope="module")
def julia():
    import hornlab_beat_bem as beat

    executable = beat.discover_julia()
    if executable is None:
        pytest.skip("no Julia executable (set HORNLAB_BEAT_JULIA)")
    return executable


def _tetrahedron(temp_dir):
    from pathlib import Path

    from hornlab_beat_bem.sweep import _WARMUP_TETRAHEDRON

    path = Path(temp_dir) / "tetra.msh"
    path.write_text(_WARMUP_TETRAHEDRON, encoding="utf-8")
    return path


def _solve(mesh_path, julia, **ground):
    import hornlab_beat_bem as beat

    return beat.solve_frequencies(
        mesh_path,
        _GROUND_FREQUENCIES,
        beat.SolveConfig(
            beat_backend="cpu",
            julia_executable=julia,
            observation=beat.ObservationConfig(
                planes=["horizontal"],
                distance_m=2.0,
                angle_min_deg=0.0,
                angle_max_deg=90.0,
                angle_count=4,
            ),
            ground_plane=ground or None,
        ),
    )


@pytest.mark.slow
def test_reported_impedance_does_not_move_6_db_when_the_ground_is_switched_on(julia):
    """The end-to-end regression for the force-scaling defect.

    A body a metre above the plane is barely loaded by its own image, so the
    physical move in reported impedance is a fraction of a dB. Scaling the
    integrated force by the image-transform count instead put it 6.02 dB high,
    with a pressure field that was entirely correct -- which is why no
    equivalence gate saw it, and why this asserts the reported number rather
    than the field.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_path = _tetrahedron(temp_dir)
        free = _solve(mesh_path, julia)
        ground = _solve(mesh_path, julia, enabled=True, axis="y", height_m=1.0)

    move_db = 20.0 * np.log10(np.abs(ground.impedance) / np.abs(free.impedance))
    assert np.all(np.isfinite(move_db))
    assert np.max(np.abs(move_db)) < 1.0, f"reported impedance moved {move_db} dB"


@pytest.mark.slow
def test_the_ground_plane_actually_changes_the_field(julia):
    """Guard against an inert image: a ground plane that did nothing would
    pass the impedance test above trivially."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_path = _tetrahedron(temp_dir)
        free = _solve(mesh_path, julia)
        ground = _solve(mesh_path, julia, enabled=True, axis="y", height_m=1.0)

    delta_db = 20.0 * np.log10(
        np.abs(ground.pressure_complex) / np.abs(free.pressure_complex)
    )
    assert np.max(np.abs(delta_db)) > 1.0


@pytest.mark.slow
def test_a_face_lying_in_the_plane_is_refused_by_the_julia_domain_guard(julia):
    """The warm-up tetrahedron has one face flat in Y=0, so at height zero it
    rests ON the plane -- that face would coincide with its own image and the
    boundary integral is singular there. Without the ported guard this solves
    in silence."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_path = _tetrahedron(temp_dir)
        with pytest.raises(RuntimeError) as excinfo:
            _solve(mesh_path, julia, enabled=True, axis="y", height_m=0.0)
    assert "lies flat" in str(excinfo.value)


@pytest.mark.slow
def test_a_body_below_the_plane_is_refused_by_the_julia_domain_guard(julia):
    """`symmetry_active_axes(:ground)` is empty, so
    `validate_symmetry_fundamental_domain!` is a no-op here and a straddling
    body would assemble against a domain that does not exist."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_path = _tetrahedron(temp_dir)
        with pytest.raises(RuntimeError) as excinfo:
            _solve(mesh_path, julia, enabled=True, axis="y", height_m=-0.02)
    assert "must lie at Y >= 0" in str(excinfo.value)


@pytest.mark.slow
def test_min_clearance_is_enforced_by_the_julia_domain_guard(julia):
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_path = _tetrahedron(temp_dir)
        with pytest.raises(RuntimeError) as excinfo:
            _solve(
                mesh_path, julia,
                enabled=True, axis="y", height_m=0.01, min_clearance_m=0.05,
            )
    assert "clearance" in str(excinfo.value)
