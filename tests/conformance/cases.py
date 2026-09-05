"""The named conformance cases.

Three, deliberately: one that scores the package's declared refusals against
the code that refuses, one that runs the ordinary exterior contract end to end
through a real Julia worker, and one absolute analytic control that a
conjugated or sign-flipped complex pressure cannot pass. Everything else in
phase 0 -- symmetry-reduced agreement, a grounded half space, an installed-wheel
baseline, multi-frequency sweeps -- belongs on this list later; the harness is
shaped so adding a case is adding a :class:`ConformanceCase`, not editing the
runner.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

import hornlab_beat_bem as beat
from hornlab_beat_bem._constants import AIR_DENSITY, SPEED_OF_SOUND
from hornlab_beat_bem.capabilities import capability_report
from hornlab_beat_bem.config import (
    GroundPlane,
    ObservationConfig,
    ObservationFrame,
    SolveConfig,
    reject_unsupported_ground_plane,
    reject_unsupported_native_symmetry,
)
from hornlab_beat_bem.sweep import (
    _WARMUP_TETRAHEDRON,
    _ground_placed_translation,
    _request_payload,
    _validated_frame_translation,
)

from .analytic import (
    icosphere,
    icosphere_surface_area,
    level_and_phase_error,
    pulsating_sphere_pressure_per_acceleration,
    write_gmsh22,
)
from .harness import ConformanceCase

#: Pulsating-sphere control geometry. These follow the defaults of
#: ``julia/scripts/validate_analytic_exterior.jl`` so the two gates report
#: discretisation error on comparable geometry: subdivision 3 is 1280 faces,
#: 22 elements per wavelength at 1 kHz, and the Julia gate measures 0.0223 dB
#: worst level error there. This case, going through the whole Python wire,
#: measures 0.0220 dB and 0.230 degrees -- the same discretisation error, which
#: is the point: the package adds a rescale and a repacking, not an error.
#: The tolerances therefore carry a factor of about 2.3 in level and 2.6 in
#: phase. They are tied to this frequency and this mesh, and raising the
#: frequency without refining the mesh will miss the phase tolerance for want
#: of elements per wavelength rather than for a regression.
SPHERE_RADIUS_M = 0.1
SPHERE_SUBDIVISIONS = 3
SPHERE_FREQUENCY_HZ = 1000.0
SPHERE_OBSERVATION_M = 3.0
SPHERE_SOURCE_TAG = 2
SPHERE_ANGLE_COUNT = 5
SPHERE_LEVEL_TOLERANCE_DB = 0.05
SPHERE_PHASE_TOLERANCE_DEG = 0.6

#: The controls must miss by a wide margin, not merely miss. Phase error is
#: measured as arg(measured/reference) and therefore wraps into (-180, 180], so
#: a conjugation's 2*k*(r-a) = 106 rad is not the number that appears: at this
#: geometry it lands at 90.4 degrees and the sign flip at 179.8, against 0.23
#: for the correct answer. Asserting a floor here rather than assuming one is
#: what keeps a future change of radius or frequency from quietly wrapping a
#: control back toward zero and leaving a gate that proves nothing.
CONTROL_MIN_PHASE_ERROR_DEG = 30.0

#: Level and phase bands for the cross-package comparison on the analytic case.
#: Both packages are scored against the same closed form, so this is a real
#: agreement check rather than a similarity impression -- but they are
#: different discretisations at different precisions, so the band is loose.
COMPARISON_LEVEL_TOLERANCE_DB = 0.5
COMPARISON_PHASE_TOLERANCE_DEG = 5.0


# ---------------------------------------------------------------------------
# Case 1: the declared refusals are the refusals the code makes
# ---------------------------------------------------------------------------


def _assert_raises(what: str, exception: type[BaseException], call) -> None:
    try:
        call()
    except exception:
        return
    except BaseException as exc:  # noqa: BLE001 - the type is the assertion
        raise AssertionError(
            f"{what}: expected {exception.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{what}: expected {exception.__name__}, nothing was raised")


def _axis_aligned_frame(origin: tuple[float, float, float]) -> ObservationFrame:
    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.asarray(origin, dtype=float),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
    )


#: Declared refusals with no Python call behind them, each with the reason it
#: cannot be exercised. The walk requires that every *other* declared refusal
#: is executed, and that every key here is still declared somewhere in the
#: report -- so an excuse cannot outlive the entry it excuses, and adding a
#: refusal to the report without either exercising it or arguing for it here
#: is a failing test rather than an unnoticed claim.
#:
#: Paths are relative to one backend's ``modes.exterior`` entry, because the
#: reason is the same on every backend.
UNEXERCISABLE_REFUSALS: dict[str, str] = {
    "formulation.refused.chief": (
        "there is no formulation switch to call: the refusal is the absence "
        "of an API. What the walk asserts instead is that the report says so "
        "-- formulation.selectable is False"
    ),
    "formulation.refused.complex_k": (
        "no formulation switch exists to call; see formulation.refused.chief"
    ),
    "formulation.refused.plain_collocation": (
        "no formulation switch exists to call; see formulation.refused.chief"
    ),
    "sources.profiles.refused.arbitrary_piston_axis": (
        "there is no piston-axis field to set wrongly: source_motion='axial' "
        "is the whole of the piston API and its axis is the global +z"
    ),
    "observation.frame.refused.frame_inference": (
        "nothing accepts a request to infer a frame, so there is no call to "
        "make; the refusal is that the argument does not exist"
    ),
    "quadrature.regular.refused_modes.wavelength": (
        "refused by the Julia driver at solve time rather than by "
        "SolveConfig, exactly as the report's own text says, so reaching it "
        "needs an accelerator worker and a device this static case must not "
        "require. The half that is checkable here is checked: SolveConfig "
        "accepts the mode, so the refusal really is the driver's"
    ),
    # The report's second spelling of a refusal, ``supported: False`` with a
    # reason. Every one of these is the absence of an API rather than a
    # rejection: there is no field to set, no argument to pass and no result
    # key to ask for, so no call can be made that the package could refuse.
    # What the walk asserts for them instead is that the report still says
    # so -- the entry is collected from the report, so deleting it here
    # without deleting it there fails, and vice versa.
    "sources.frequency_dependent_drive": (
        "no per-frequency drive callback or weight field exists on "
        "SolveConfig, so there is nothing to hand a frequency-dependent "
        "drive to; velocity_sources carries one amplitude for the whole sweep"
    ),
    "observation.arbitrary_points": (
        "ObservationConfig takes a distance and an angle grid; there is no "
        "points argument to fill with arbitrary coordinates, so the refusal "
        "is that the parameter does not exist"
    ),
    "observation.post_solve_field_replay": (
        "surface_traces=True is accepted and returns the traces; what is "
        "missing is an exterior evaluator to feed them to, and this package "
        "exports no such function to call"
    ),
    "quantities.acoustic_power": (
        "SolveResult has no power field and no integrator produces one, so "
        "the absence is checked by reading the result contract rather than "
        "by making a call that raises"
    ),
    "quantities.per_tag_mean_pressures": (
        "SolveResult carries one impedance array, not a per-tag mapping; "
        "with one drivable tag (sources.count) there is no second mean "
        "pressure to ask for and no argument that would ask for it"
    ),
    "mesh_input.multiple_bodies": (
        "solve_frequencies takes one mesh path; a second body would need a "
        "second argument that does not exist, so nothing can be passed for "
        "the request compiler to refuse"
    ),
}

#: Exercisers for refusals the report spells as ``supported: False`` with a
#: reason, keyed by the path :func:`_declared_refusal_paths` collects. A path
#: with no entry here must be excused in :data:`UNEXERCISABLE_REFUSALS`; the
#: walk asserts that, so this table cannot quietly shrink.
#:
#: It exists so the near-correction refusal is driven by the *collected* path
#: rather than by a hand-written branch -- the pattern this case exists to
#: enforce, and the recorded verified path is then one the report declares.
_UNSUPPORTED_FEATURE_CHECKS: dict[str, Any] = {
    "quadrature.near_pair": lambda backend, config: _assert_raises(
        f"{backend}: near_correction",
        NotImplementedError,
        lambda: config(near_correction=True),
    ),
}

#: Orders the walk feeds ``quadrature_order`` to prove the refusal bites. 6 is
#: the one that matters: before this was validated it was accepted and solved
#: with the 3-point rule, i.e. *less* accurately than the default.
_UNSUPPORTED_QUADRATURE_ORDERS = (3, 6, 8)

#: Frequency lists the sweep entry point must refuse, by the report's own key.
_REFUSED_FREQUENCY_LISTS: dict[str, list[float]] = {
    "empty": [],
    "non_finite": [1000.0, float("nan")],
    "non_positive": [1000.0, -250.0],
    "duplicate": [1000.0, 1000.0],
}


def _declared_unsupported_paths(exterior: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The report's *other* spelling of a refusal: ``supported: False`` + reason.

    Returned as a mapping so a caller can read the reason back. A
    ``supported: False`` entry with no ``reason`` is deliberately **not**
    collected -- the report's own rule is that an unsupported capability is
    reported with the reason it is unsupported, and a bare False would be
    excused into invisibility by being collected here.
    """

    found: dict[str, dict[str, Any]] = {}

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if path and node.get("supported") is False and node.get("reason"):
            found[path] = node
        for key, value in node.items():
            walk(value, f"{path}.{key}" if path else key)

    walk(exterior, "")
    return found


def _declared_refusal_paths(exterior: dict[str, Any]) -> set[str]:
    """Every refusal one backend's exterior entry declares, found by shape.

    Collected rather than listed, so a refusal added to the report is walked
    without anyone remembering to add it here -- the failure mode of the first
    version of this case, which walked six of the declared blocks and left
    six unwalked. The rules:

    * a key named ``refused``, ``refused_*`` or ``*_refused`` declares one
      refusal per sub-key when it maps to a dictionary, and one refusal at its
      own path when it is prose (``sources.count.refused_note``);
    * ``supported: False`` beside a ``reason`` is the report's second spelling
      of a refusal (``quadrature.near_pair`` on the accelerators,
      ``observation.arbitrary_points`` everywhere) and declares one refusal at
      its own path -- see :func:`_declared_unsupported_paths`. Leaving this
      shape out is what let ``near_correction`` be walked by a hand-written
      branch recording a path the report never declared;
    * ``requires_frame_override: True`` is the one constraint the report
      spells as a requirement rather than as a refusal, and is treated as one.
    """

    found: set[str] = set(_declared_unsupported_paths(exterior))

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            declares = (
                key == "refused" or key.startswith("refused_") or key.endswith("_refused")
            )
            if declares and isinstance(value, dict):
                found.update(f"{here}.{name}" for name in value)
            elif declares:
                found.add(here)
            elif key == "requires_frame_override" and value is True:
                found.add(here)
            else:
                walk(value, here)

    walk(exterior, "")
    return found


def _verify_declared_refusals(
    backend: str, exterior: dict[str, Any]
) -> tuple[set[str], list[str]]:
    """Exercise one backend's declared refusals; return what was proven.

    Every construction goes through ``beat_backend=backend`` so the refusals
    are checked on the backend that declares them rather than on the CPU entry
    standing in for all four.
    """

    verified: set[str] = set()
    supported: list[str] = []

    def config(**kwargs: Any) -> SolveConfig:
        return SolveConfig(beat_backend=backend, **kwargs)

    # -- formulation: nothing to call, so assert the claim that makes it so
    assert exterior["formulation"]["selectable"] is False, (
        "the formulation refusals are excused on the ground that no switch "
        "exists; a selectable formulation would need them exercised instead"
    )

    # -- symmetry
    for plane in exterior["symmetry"]["refused"]:
        _assert_raises(
            f"{backend}: native_symmetry_plane={plane!r}",
            NotImplementedError,
            lambda plane=plane: reject_unsupported_native_symmetry(
                config(native_symmetry_plane=plane)
            ),
        )
        verified.add(f"symmetry.refused.{plane}")
    for name, entry in exterior["symmetry"]["supported"].items():
        built = config(native_symmetry_plane=entry["value"])
        reject_unsupported_native_symmetry(built)
        assert beat.beat_image_mode(built) == entry["solver_image_mode"], (
            f"symmetry {name!r} maps to {beat.beat_image_mode(built)!r}, "
            f"not the reported {entry['solver_image_mode']!r}"
        )
        supported.append(f"symmetry:{name}")

    # -- ground plane, its axes and its two refused compositions
    for axis in exterior["ground_plane"]["refused_axes"]:
        _assert_raises(
            f"{backend}: ground_plane.axis={axis!r}",
            NotImplementedError,
            lambda axis=axis: reject_unsupported_ground_plane(
                config(ground_plane=GroundPlane(enabled=True, axis=axis))
            ),
        )
        verified.add(f"ground_plane.refused_axes.{axis}")
    for axis in exterior["ground_plane"]["supported_axes"]:
        built = config(ground_plane=GroundPlane(enabled=True, axis=axis))
        reject_unsupported_ground_plane(built)
        assert beat.beat_image_mode(built) == exterior["ground_plane"]["solver_image_mode"]
        supported.append(f"ground_axis:{axis}")

    _assert_raises(
        f"{backend}: ground plane combined with native symmetry",
        NotImplementedError,
        lambda: reject_unsupported_ground_plane(
            config(
                ground_plane=GroundPlane(enabled=True, axis="y"),
                native_symmetry_plane="yz",
            )
        ),
    )
    verified.add("ground_plane.combinations_refused.ground_plane+native_symmetry")

    # A frame origin off the plane, through the real frame validator, so what
    # is refused is the translation the solve would actually have applied.
    _assert_raises(
        f"{backend}: a frame origin off the ground plane",
        NotImplementedError,
        lambda: _ground_placed_translation(
            _validated_frame_translation(_axis_aligned_frame((0.0, 0.25, 0.0))),
            config(ground_plane=GroundPlane(enabled=True, axis="y")),
        ),
    )
    verified.add(
        "ground_plane.combinations_refused.ground_plane+frame_translation_along_axis"
    )

    # -- observation frame and named origin
    _assert_raises(
        f"{backend}: observation.origin without frame_override",
        NotImplementedError,
        lambda: config(observation=ObservationConfig(origin="mouth")),
    )
    verified.add("observation.named_origin.requires_frame_override")
    config(
        observation=ObservationConfig(origin="mouth"),
        frame_override=_axis_aligned_frame((0.0, 0.0, 0.32)),
    )
    supported.append("observation:named_origin_with_frame")

    _assert_raises(
        f"{backend}: tilted observation frame",
        NotImplementedError,
        lambda: _validated_frame_translation(
            ObservationFrame(
                axis=np.array([0.1, 0.0, 0.99498743710662]),
                origin=np.zeros(3),
                u=np.array([1.0, 0.0, 0.0]),
                v=np.array([0.0, 1.0, 0.0]),
            )
        ),
    )
    verified.add("observation.frame.refused.tilted_frame")

    # -- sources: one tag, unit amplitude, real amplitude
    _assert_raises(
        f"{backend}: two source tags",
        NotImplementedError,
        lambda: config(velocity_sources={2: 1.0, 3: 1.0}),
    )
    verified.add("sources.count.refused_note")
    _assert_raises(
        f"{backend}: non-unit source amplitude",
        NotImplementedError,
        lambda: config(velocity_sources={2: 0.5}),
    )
    _assert_raises(
        f"{backend}: complex source amplitude",
        NotImplementedError,
        lambda: config(velocity_sources={2: 1.0 + 0.5j}),
    )
    verified.add("sources.amplitude.refused_note")

    for motion in exterior["sources"]["profiles"]["supported"]:
        config(source_motion=motion)
        supported.append(f"source_motion:{motion}")
    for motion in exterior["sources"]["profiles"]["refused"]:
        path = f"sources.profiles.refused.{motion}"
        if path in UNEXERCISABLE_REFUSALS:
            continue
        _assert_raises(
            f"{backend}: source_motion={motion!r}",
            ValueError,
            lambda motion=motion: config(source_motion=motion),
        )
        verified.add(path)

    # -- frequency list, refused at the sweep entry point before any worker
    for name, values in _REFUSED_FREQUENCY_LISTS.items():
        _assert_raises(
            f"{backend}: {name} frequency list",
            ValueError,
            lambda values=values: beat.solve_frequencies("mesh.msh", values, config()),
        )
        verified.add(f"frequencies.explicit_list.refused.{name}")
    # The order claim, on the half that is reachable without a solver: the
    # request carries the list as given, not sorted.
    unordered = [2000.0, 500.0, 1000.0]
    payload = _request_payload(
        "mesh.msh", np.asarray(unordered, dtype=float), config(), (0.0, 0.0, 0.0)
    )
    assert payload["frequencies_hz"] == unordered, (
        "the report declares frequencies_hz order-preserving, but the request "
        f"carries {payload['frequencies_hz']}"
    )
    supported.append("frequencies:order_preserved_in_the_request")

    # -- precision
    if "double" in exterior["precision"]["requested_values"]:
        config(solve_precision="double")
        supported.append("precision:double")
    else:
        _assert_raises(
            f"{backend}: solve_precision='double'",
            NotImplementedError,
            lambda: config(solve_precision="double"),
        )
        verified.add("precision.refused.double")

    # -- quadrature: the orders the vendored rule really distinguishes
    regular = exterior["quadrature"]["regular"]
    for order in regular["supported_orders"]:
        config(quadrature_order=order)
        supported.append(f"quadrature_order:{order}")
    for order in _UNSUPPORTED_QUADRATURE_ORDERS:
        assert order not in regular["supported_orders"], (
            f"order {order} is advertised as supported, so it cannot be used "
            "to prove the refusal"
        )
        _assert_raises(
            f"{backend}: quadrature_order={order}",
            ValueError,
            lambda order=order: config(quadrature_order=order),
        )
    verified.add("quadrature.regular.refused_orders_note")

    for mode in regular["supported_modes"]:
        config(regular_quadrature_mode=mode)
        supported.append(f"regular_quadrature_mode:{mode}")
    for mode in regular.get("refused_modes", {}):
        # Declared as a solve-time refusal in the Julia driver. Assert the
        # half that is reachable: SolveConfig does *not* also refuse it, so
        # the report names the right refusing party.
        config(regular_quadrature_mode=mode)
        supported.append(f"regular_quadrature_mode_constructs:{mode}")

    # -- near-singular correction, on the backends that have it
    near = exterior["quadrature"]["near_pair"]
    if near["supported"]:
        config(near_correction=True)
        supported.append("near_correction")
    else:
        assert near.get("reason"), (
            "a refused near_pair entry must say why: without a reason it is "
            "not collected as a declared refusal and nothing would walk it"
        )

    # -- the refusals spelled ``supported: False`` + reason, driven by the
    # paths the collector found rather than by a branch per feature. The one
    # with a callable surface (near_pair) is exercised here, so the path this
    # records is a path the report declares; the rest are the absence of an
    # API and are excused, which the caller's walk enforces.
    for path in sorted(_declared_unsupported_paths(exterior)):
        check = _UNSUPPORTED_FEATURE_CHECKS.get(path)
        if check is None:
            continue
        check(backend, config)
        verified.add(path)

    return verified, supported


def _check_declared_refusals() -> dict[str, Any]:
    """Every refusal the report declares, exercised against the code that refuses.

    A capability report is a liability the moment it drifts from the code, and
    the drift that matters is the report saying "refused" over something the
    package quietly accepts -- which is precisely the B2 defect, an accepted
    option with no effect.

    So the declared refusals are *collected from the report itself*
    (:func:`_declared_refusal_paths`, in both of the shapes the report uses to
    spell a refusal) rather than transcribed, and every one of them must
    either be exercised on the backend that declares it or carry an entry in
    :data:`UNEXERCISABLE_REFUSALS` saying why no call can reach it. A refusal
    added to the report is therefore walked by construction, and a stale
    excuse -- an allow-list entry whose report entry has gone -- fails too.

    The counts returned are path-instances: one declared refusal on one
    backend. Declared equals exercised plus excused, and that identity is
    asserted rather than described.
    """

    report = capability_report()
    declared_paths: set[str] = set()
    declared: list[str] = []
    verified: list[str] = []
    excused: list[str] = []
    modes_verified: list[str] = []
    supported: list[str] = []

    for backend, backend_entry in sorted(report["backends"].items()):
        exterior = backend_entry["modes"]["exterior"]
        paths = _declared_refusal_paths(exterior)
        declared_paths |= paths
        declared.extend(f"{backend}.{path}" for path in paths)

        proven, constructed = _verify_declared_refusals(backend, exterior)
        verified.extend(f"{backend}.{path}" for path in proven)
        supported.extend(f"{backend}:{label}" for label in constructed)
        excused.extend(
            f"{backend}.{path}"
            for path in paths
            if path not in proven and path in UNEXERCISABLE_REFUSALS
        )

        unwalked = sorted(
            path
            for path in paths
            if path not in proven and path not in UNEXERCISABLE_REFUSALS
        )
        assert not unwalked, (
            f"{backend}: the report declares refusals nothing exercised: "
            + ", ".join(unwalked)
            + " -- exercise them, or record why they cannot be in "
            "UNEXERCISABLE_REFUSALS"
        )

        for mode, entry in backend_entry["modes"].items():
            if mode == "exterior":
                continue
            assert entry["supported"] is False and entry["reason"], (
                f"mode {mode!r} claims support without a reachable API"
            )
            modes_verified.append(f"{backend}.modes.{mode}.unsupported")

    stale = sorted(
        (set(UNEXERCISABLE_REFUSALS) | set(_UNSUPPORTED_FEATURE_CHECKS))
        - declared_paths
    )
    assert not stale, (
        "UNEXERCISABLE_REFUSALS or _UNSUPPORTED_FEATURE_CHECKS covers "
        "refusals the report no longer declares: " + ", ".join(stale)
    )

    # One unit throughout: a *path-instance* is one declared refusal on one
    # backend, so the same path on four backends counts four times. Mixing
    # instances with distinct paths is how the first version of these numbers
    # came out adding to more than the total.
    assert len(declared) == len(verified) + len(excused), (
        f"{len(declared)} declared path-instances is not {len(verified)} "
        f"exercised + {len(excused)} excused"
    )

    return {
        "unit": "path-instances: one declared refusal on one backend",
        "refusals_declared": sorted(declared),
        "refusals_verified": sorted(verified),
        "refusals_excused": sorted(excused),
        "refusals_unexercisable": sorted(UNEXERCISABLE_REFUSALS),
        "declared_count": len(declared),
        "refusal_count": len(verified),
        "excused_count": len(excused),
        "unexercisable_path_count": len(UNEXERCISABLE_REFUSALS),
        "unsupported_modes_verified": sorted(modes_verified),
        "unsupported_mode_count": len(modes_verified),
        "supported_combinations_constructed": sorted(supported),
        "supported_count": len(supported),
    }


# ---------------------------------------------------------------------------
# Case 2: the ordinary exterior contract, end to end
# ---------------------------------------------------------------------------


def _tetrahedron_mesh(work_dir: Path) -> Path:
    path = work_dir / "warmup_tetrahedron.msh"
    path.write_text(_WARMUP_TETRAHEDRON, encoding="utf-8")
    return path


def _tetrahedron_config(julia: str) -> SolveConfig:
    return SolveConfig(
        beat_backend=beat.BEAT_CPU,
        julia_executable=julia,
        observation=ObservationConfig(
            planes=["horizontal", "vertical"],
            distance_m=1.0,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=4,
        ),
    )


def _accept_exterior_contract(result: beat.SolveResult) -> dict[str, Any]:
    frequencies = np.asarray(result.frequencies_hz)
    assert result.requested_frequency_count == frequencies.size
    assert not result.cancelled and not result.is_partial
    assert result.pressure_complex.shape == (frequencies.size, 2, 4)
    assert np.all(np.isfinite(result.pressure_complex))
    assert np.all(np.isfinite(result.impedance))

    # spl_db is absolute and directivity_db is the same field normalized, so
    # their difference is one number per frequency and plane -- the B1 contract
    # measured on a real solve rather than on a hand-built result.
    offsets = result.spl_db - result.directivity_db
    spread = float(np.max(np.abs(offsets - offsets[:, :, :1])))
    assert spread < 1e-9, f"directivity is not a per-cut offset of SPL ({spread})"
    reference = result.directivity_reference_index
    assert np.allclose(result.directivity_db[:, :, reference], 0.0, atol=1e-12)

    # impedance is a mean pressure per unit normal acceleration; multiplying
    # by -i*omega puts it back on a velocity basis, whose real part is the
    # radiation resistance and must be positive for a source that radiates.
    omega = 2.0 * np.pi * frequencies
    specific = -1j * omega * result.impedance
    assert np.all(specific.real > 0.0), "a radiating source must show positive resistance"
    return {
        "directivity_reference_deg": result.directivity_reference_deg,
        "spl_minus_directivity_db": [float(value) for value in offsets[:, 0, 0]],
        "on_axis_spl_db": [float(value) for value in result.spl_db[:, 0, 0]],
        "radiation_resistance_pa_s_per_m": [float(value) for value in specific.real],
        "max_directivity_offset_spread_db": spread,
    }


# ---------------------------------------------------------------------------
# Case 3: the phase-sensitive analytic control
# ---------------------------------------------------------------------------


def _sphere_mesh(work_dir: Path) -> Path:
    vertices, faces = icosphere(SPHERE_RADIUS_M, SPHERE_SUBDIVISIONS)
    return write_gmsh22(
        work_dir / "pulsating_sphere.msh",
        vertices,
        faces,
        physical_tag=SPHERE_SOURCE_TAG,
        physical_name="sphere",
    )


def _sphere_observation() -> ObservationConfig:
    return ObservationConfig(
        planes=["horizontal", "vertical"],
        distance_m=SPHERE_OBSERVATION_M,
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=SPHERE_ANGLE_COUNT,
    )


def _sphere_config(julia: str) -> SolveConfig:
    return SolveConfig(
        beat_backend=beat.BEAT_CPU,
        julia_executable=julia,
        # Float64 assembly and solve, so what the tolerances measure is
        # discretisation rather than single-precision noise. The wire is still
        # Float32 -- see the capability report's precision entry.
        solve_precision="double",
        velocity_sources={SPHERE_SOURCE_TAG: 1.0},
        source_motion="normal",
        observation=_sphere_observation(),
        air_density=AIR_DENSITY,
        sound_speed=SPEED_OF_SOUND,
    )


def sphere_reference_pressure() -> complex:
    return pulsating_sphere_pressure_per_acceleration(
        SPHERE_OBSERVATION_M,
        SPHERE_RADIUS_M,
        SPHERE_FREQUENCY_HZ,
        air_density=AIR_DENSITY,
        sound_speed=SPEED_OF_SOUND,
    )


def _accept_analytic_sphere(result: beat.SolveResult) -> dict[str, Any]:
    """Score the measured field against the closed form, then against two lies.

    The controls are the point. A uniformly breathing sphere driven by a real
    normal velocity produces a field whose conjugate has an identical
    magnitude everywhere, so a level tolerance -- however tight -- cannot tell
    a conjugated result from a correct one. Only the phase can, and the two
    controls here assert that it does: the same measurement scored against
    ``conj(p)`` and against ``-p`` must miss by a wide margin.
    """

    reference = sphere_reference_pressure()
    measured = np.asarray(result.pressure_complex)
    correct = level_and_phase_error(measured, reference)
    assert correct["worst_level_error_db"] < SPHERE_LEVEL_TOLERANCE_DB, (
        f"level error {correct['worst_level_error_db']:.4f} dB exceeds "
        f"{SPHERE_LEVEL_TOLERANCE_DB} dB"
    )
    assert correct["worst_phase_error_deg"] < SPHERE_PHASE_TOLERANCE_DEG, (
        f"phase error {correct['worst_phase_error_deg']:.4f} deg exceeds "
        f"{SPHERE_PHASE_TOLERANCE_DEG} deg"
    )

    conjugated = level_and_phase_error(measured, np.conjugate(reference))
    sign_flipped = level_and_phase_error(measured, -reference)
    for label, control in (("conjugated", conjugated), ("sign_flipped", sign_flipped)):
        assert control["worst_phase_error_deg"] > CONTROL_MIN_PHASE_ERROR_DEG, (
            f"the {label} control is only {control['worst_phase_error_deg']:.4f} "
            "deg away, so this case is not phase-sensitive"
        )
    # The conjugation control's whole reason for existing: it is invisible to
    # a magnitude-only gate, so assert that it is, rather than assuming it.
    level_gap = abs(
        conjugated["worst_level_error_db"] - correct["worst_level_error_db"]
    )
    assert level_gap <= 1e-12, (
        "the conjugated control differs in level by "
        f"{level_gap:.3e} dB, so it is not the level-invisible control it claims"
    )

    # A breathing sphere is isotropic: every cut and every angle sees the same
    # magnitude, which a transposed or mis-stacked (F, P, N) block would break.
    amplitudes = np.abs(measured)
    isotropy_db = float(
        20.0 * np.log10(np.max(amplitudes) / np.min(amplitudes))
    )
    assert isotropy_db < SPHERE_LEVEL_TOLERANCE_DB, (
        f"the field is not isotropic across the cuts ({isotropy_db:.4f} dB)"
    )
    return {
        "reference_pressure": {
            "real": float(reference.real),
            "imag": float(reference.imag),
            "abs": float(abs(reference)),
            "arg_deg": float(math.degrees(np.angle(reference))),
        },
        "against_closed_form": correct,
        "control_conjugated": conjugated,
        "control_sign_flipped": sign_flipped,
        "isotropy_spread_db": isotropy_db,
        "faceted_area_m2": icosphere_surface_area(
            SPHERE_RADIUS_M, SPHERE_SUBDIVISIONS
        ),
        "analytic_area_m2": 4.0 * math.pi * SPHERE_RADIUS_M**2,
    }


def _metal_bem_sphere_config() -> Any:
    """The hornlab-metal-bem counterpart of :func:`_sphere_config`.

    Imported lazily and used read-only. The frame is passed explicitly rather
    than inferred: metal-bem infers an observation frame from the source tag's
    geometry, and on a closed sphere that inference has no throat to find, so
    an inferred frame would silently observe a different arc from BEAT's.
    """

    import hornlab_metal_bem as metal

    zero = np.zeros(3)
    frame = metal.ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=zero.copy(),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=zero.copy(),
        source_center=zero.copy(),
    )
    return metal.native_config(
        velocity_sources={SPHERE_SOURCE_TAG: 1.0},
        observation=metal.ObservationConfig(
            planes=["horizontal", "vertical"],
            distance_m=SPHERE_OBSERVATION_M,
            angle_min_deg=0.0,
            angle_max_deg=180.0,
            angle_count=SPHERE_ANGLE_COUNT,
        ),
        frame_override=frame,
        air_density=AIR_DENSITY,
    )


def _compare_sphere(
    beat_result: beat.SolveResult,
    metal_result: Any,
    comparison: dict[str, Any],
) -> None:
    """Score both packages against the same closed form, and against each other.

    The bands are asserted here rather than only in the pytest reader, so the
    standalone ``python -m conformance`` run fails on a disagreement instead of
    recording one and printing PASS. ``run_case`` calls this inside its guarded
    section, so a failure here is the case's failure and its record still gets
    written.

    ``comparison`` is the record's own comparison entry, and the scored table
    is assigned into it **before** the bands are asserted. Building it in a
    local and returning it at the end is what this signature exists to
    prevent: on a band violation the return never happened, so the record said
    ``ran: true`` and carried no numbers -- exactly the evidence a
    disagreement has to be argued from.
    """

    reference = sphere_reference_pressure()
    beat_pressure = np.asarray(beat_result.pressure_complex)
    metal_pressure = np.asarray(metal_result.pressure_complex)
    metrics: dict[str, Any] = {
        "beat_vs_closed_form": level_and_phase_error(beat_pressure, reference),
        "metal_bem_vs_closed_form": level_and_phase_error(metal_pressure, reference),
        "shapes": {
            "beat": list(beat_pressure.shape),
            "metal_bem": list(metal_pressure.shape),
        },
    }
    if beat_pressure.shape == metal_pressure.shape:
        metrics["beat_vs_metal_bem"] = level_and_phase_error(
            beat_pressure, metal_pressure
        )
    metrics["agreement_band"] = {
        "level_db": COMPARISON_LEVEL_TOLERANCE_DB,
        "phase_deg": COMPARISON_PHASE_TOLERANCE_DEG,
    }
    # In the record before anything is asserted, so a violation is recorded
    # with the numbers that show it rather than with an error string alone.
    comparison["metrics"] = metrics
    for label in ("metal_bem_vs_closed_form", "beat_vs_metal_bem"):
        scored = metrics.get(label)
        if scored is None:
            continue
        assert scored["worst_level_error_db"] < COMPARISON_LEVEL_TOLERANCE_DB, (
            f"{label}: level error {scored['worst_level_error_db']:.4f} dB "
            f"exceeds the {COMPARISON_LEVEL_TOLERANCE_DB} dB band"
        )
        assert scored["worst_phase_error_deg"] < COMPARISON_PHASE_TOLERANCE_DEG, (
            f"{label}: phase error {scored['worst_phase_error_deg']:.4f} deg "
            f"exceeds the {COMPARISON_PHASE_TOLERANCE_DEG} deg band"
        )


# ---------------------------------------------------------------------------


CAPABILITY_REFUSALS = ConformanceCase(
    name="capability_refusals",
    description=(
        "every refusal the capability report declares is a refusal the package "
        "actually makes, and every supported combination constructs"
    ),
    static_check=_check_declared_refusals,
    tags=("capability", "contract", "B2"),
)

EXTERIOR_CONTRACT = ConformanceCase(
    name="exterior_contract_two_frequencies",
    description=(
        "an ordinary CPU exterior solve: completion status, result shapes, "
        "absolute SPL against normalized directivity, positive radiation "
        "resistance"
    ),
    frequencies_hz=(300.0, 500.0),
    make_mesh=_tetrahedron_mesh,
    make_config=_tetrahedron_config,
    accept=_accept_exterior_contract,
    tags=("exterior", "contract", "B1", "B3"),
    parameters={"mesh": "4-triangle tetrahedron", "solve_precision": "single"},
)

ANALYTIC_SPHERE_PHASE = ConformanceCase(
    name="analytic_pulsating_sphere_phase",
    description=(
        "absolute analytic control: a breathing sphere's complex pressure "
        "against the closed form, with conjugated and sign-flipped controls "
        "that must fail"
    ),
    frequencies_hz=(SPHERE_FREQUENCY_HZ,),
    make_mesh=_sphere_mesh,
    make_config=_sphere_config,
    accept=_accept_analytic_sphere,
    make_metal_bem_config=_metal_bem_sphere_config,
    compare=_compare_sphere,
    tags=("analytic", "phase", "convention"),
    parameters={
        "sphere_radius_m": SPHERE_RADIUS_M,
        "icosphere_subdivisions": SPHERE_SUBDIVISIONS,
        "observation_radius_m": SPHERE_OBSERVATION_M,
        "solve_precision": "double",
        "level_tolerance_db": SPHERE_LEVEL_TOLERANCE_DB,
        "phase_tolerance_deg": SPHERE_PHASE_TOLERANCE_DEG,
    },
)


def all_cases() -> list[ConformanceCase]:
    return [CAPABILITY_REFUSALS, EXTERIOR_CONTRACT, ANALYTIC_SPHERE_PHASE]


def case_by_name(name: str) -> ConformanceCase:
    for case in all_cases():
        if case.name == name:
            return case
    raise KeyError(f"unknown conformance case: {name!r}")
