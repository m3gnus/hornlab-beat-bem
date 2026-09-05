"""What this package supports, and what it refuses, per backend and mode.

A machine-readable capability contract. It answers one question -- *what can a
caller ask this package for at this commit* -- and deliberately does not answer
a second one: whether this host can run it. That is
:func:`hornlab_beat_bem.beat_engine_status`, which probes hardware and a Julia
runtime and therefore changes underneath a caller. Keeping the two apart is
what lets this report be compared between machines, pinned in a consumer's
test, and diffed across commits.

The rule the entries follow is that a capability which exists in the vendored
Julia engine but has no path through the supported Python API is **not** a
package capability. The engine can solve coupled FEM-BEM-LEM problems and this
package cannot ask it to, so ``coupled_fem_bem_lem`` is reported unsupported
with that as its reason rather than omitted or reported true. The same applies
to multiple source channels, arbitrary observation points, and post-solve field
replay.

Versioning: :data:`CAPABILITY_SCHEMA_VERSION` moves when the *shape* changes.
A capability flipping from false to true is a content change and does not move
it; removing a field or renaming one does. A consumer that reads this should
check the schema version and refuse a shape it does not know, rather than
assuming a missing key means "unsupported".
"""

from __future__ import annotations

from typing import Any

from ._constants import AIR_DENSITY, REFERENCE_PRESSURE, SPEED_OF_SOUND
from .config import (
    BEAT_BACKENDS,
    BEAT_CPU,
    BEAT_CUDA,
    BEAT_METAL,
    BEAT_ROCM,
    GROUND_PLANE_AXES,
    NEAR_CORRECTION_REFUSALS,
    SUPPORTED_QUADRATURE_ORDERS,
    SolveConfig,
)

#: Shape version of :func:`capability_report`. Bump on a structural change.
#:
#: 2 (2026-09-05): ``quadrature.regular`` gained ``supported_orders`` and
#: ``refused_orders_note``, and the exterior mode gained a ``frequencies``
#: block. Both are shape changes, so the version moves; the Metal near-pair
#: entry flipping to unsupported in the same commit is a content change and
#: would not have moved it on its own.
CAPABILITY_SCHEMA_VERSION = 2

#: Wire schema version of the request ``sweep._request_payload`` builds. Kept
#: here so a capability consumer sees the request contract it is paired with.
REQUEST_SCHEMA_VERSION = 2

#: Solve modes this report enumerates for every backend. Only ``exterior`` is
#: reachable from the Python API today; the rest are listed so that "absent"
#: is a recorded answer with a reason rather than a gap in the report.
SOLVE_MODES = (
    "exterior",
    "exterior_robin",
    "infinite_baffle",
    "circsym_m0",
    "coupled_fem_bem_lem",
)

_BACKEND_PROJECTS = {
    BEAT_CPU: "hornlab_beat_bem/julia",
    BEAT_CUDA: "hornlab_beat_bem/julia_cuda",
    BEAT_ROCM: "hornlab_beat_bem/julia_rocm",
    BEAT_METAL: "hornlab_beat_bem/julia_metal",
}

#: Read, not restated: ``SolveConfig`` refuses exactly these backends with
#: exactly these reasons, so the report cannot promise a correction the solve
#: does not run. That drift is the defect this whole module is against.
_NEAR_CORRECTION_REFUSALS = NEAR_CORRECTION_REFUSALS


def _conventions() -> dict[str, Any]:
    return {
        "length_unit": "m",
        "time_convention": "exp(-i*omega*t)",
        "outgoing_wave": "exp(+i*k*r)",
        "drive_quantity": "normal acceleration",
        "drive_amplitude": 1.0,
        "drive_amplitude_unit": "m/s^2",
        "drive_basis_note": (
            "the vendored solver drives q = i*rho*omega*v_n on a 1 m/s normal "
            "velocity basis; every array this package returns is rescaled by "
            "1/(-i*omega) to the shared unit-normal-acceleration convention"
        ),
        "spl_reference_pressure_pa": REFERENCE_PRESSURE,
        "medium_defaults": {
            "air_density_kg_m3": AIR_DENSITY,
            "sound_speed_m_s": SPEED_OF_SOUND,
            "configurable": True,
        },
    }


def _formulation(backend: str) -> dict[str, Any]:
    return {
        "name": "burton_miller",
        "coupling": "i/k",
        "operator": "(0.5 M - D + (i/k) H) p = -(S + (i/k)(K' + 0.5 M_p1dp0)) q",
        "trial_spaces": {"pressure": "P1 per vertex", "neumann": "DP0 per face"},
        "selectable": False,
        "selectable_note": (
            "the exterior solve is always Burton-Miller; the package exposes "
            "no formulation switch"
        ),
        "refused": {
            "chief": "no CHIEF point machinery is reachable from this package",
            "complex_k": "no complex-wavenumber damping option is exposed",
            "plain_collocation": (
                "an uncoupled single/double layer solve is not selectable"
            ),
        },
    }


def _precision(backend: str) -> dict[str, Any]:
    requested = ["single", "double"] if backend == BEAT_CPU else ["single"]
    entry: dict[str, Any] = {
        "requested_values": requested,
        "default": "single",
        "actual_assembly_and_solve": {"single": "float32", "double": "float64"},
        "result_serialization": "float32",
        "result_serialization_note": (
            "the Julia driver serialises complex pressures, SPL and surface "
            "traces as Float32 on every backend and in both precisions, so a "
            "'double' solve is Float64 internally and Float32 on the wire; "
            "this is a precision-contract limit, not a solve setting"
        ),
        "solved_frequency_precision": "float32",
        "solved_frequency_note": (
            "the driver casts the requested frequency list to Float32 before "
            "solving; the returned frequency axis is the requested Float64 "
            "value, cross-checked against the Float32 echo to 1e-6 relative"
        ),
        "returned_array_dtype": {
            "pressure_complex": "complex128",
            "spl_db": "float64",
            "frequencies_hz": "float64",
        },
        "fallback_precision": "same-precision LU; no Float64 refinement step",
    }
    if backend != BEAT_CPU:
        entry["refused"] = {
            "double": (
                "solve_precision='double' is available on the BEAT CPU "
                "backend only; the accelerator kernels are Float32"
            )
        }
    return entry


def _symmetry() -> dict[str, Any]:
    return {
        "field": "native_symmetry_plane",
        "supported": {
            "none": {
                "value": None,
                "solver_image_mode": "off",
                "image_transform_count": 1,
                "physical_radiator_count": 1,
            },
            "yz": {
                "value": "yz",
                "solver_image_mode": "x",
                "image_transform_count": 2,
                "physical_radiator_count": 2,
                "note": "half domain, mirrored across x = 0",
            },
            "yz+xz": {
                "value": "yz+xz",
                "solver_image_mode": "xy",
                "image_transform_count": 4,
                "physical_radiator_count": 4,
                "note": "quarter domain, mirrored across x = 0 and y = 0",
            },
        },
        "refused": {
            "xz": (
                "a y-only half domain: the BEAT Engine mirrors across x = 0 "
                "or across both x = 0 and y = 0, and has no y-only mirror"
            ),
            "xy": (
                "legacy plane token; 'xy' names three different things across "
                "this workspace, so it is refused rather than guessed at"
            ),
        },
        "reduced_mesh_domain": "positive fundamental domain",
    }


def _ground_plane() -> dict[str, Any]:
    return {
        "field": "ground_plane",
        "supported_axes": list(GROUND_PLANE_AXES),
        "refused_axes": {
            "x": "a side wall beside the horn; BEAT mirrors across y = 0 only",
            "z": "a rigid wall behind the throat; BEAT mirrors across y = 0 only",
        },
        "plane": "y = 0, fluid occupies y >= 0",
        "solver_image_mode": "ground",
        "image_transform_count": 2,
        "physical_radiator_count": 1,
        "physical_radiator_note": (
            "a rigid half space's image is fictitious, so the reported "
            "impedance counts the source once; counting the image as a "
            "radiator is a 6.02 dB error over an entirely correct field"
        ),
        "placement": {
            "field": "ground_plane.height_m",
            "meaning": "height of the model origin above the plane",
            "containment_tolerance_m": 1.0e-6,
            "min_clearance_field": "ground_plane.min_clearance_m",
        },
        "combinations_refused": {
            "ground_plane+native_symmetry": (
                "the solver carries a single image-transform set, so a "
                "grounded solve runs the full domain; the combination is "
                "refused rather than letting one mode quietly win"
            ),
            "ground_plane+frame_translation_along_axis": (
                "the ground placement owns its axis; a frame origin off the "
                "plane is refused rather than overriding the placement"
            ),
        },
    }


def _sources() -> dict[str, Any]:
    return {
        "count": {
            "supported": 1,
            "refused_note": (
                "the engine has multiple source channels; the package request "
                "compiler drives exactly one tag and one synthesised channel"
            ),
        },
        "amplitude": {
            "supported": [1.0],
            "refused_note": "non-unit and complex drive amplitudes are refused",
        },
        "frequency_dependent_drive": {
            "supported": False,
            "reason": "no per-frequency drive callback or weight is exposed",
        },
        "profiles": {
            "field": "source_motion",
            "supported": {
                "normal": "uniform outward normal velocity on the source tag",
                "axial": (
                    "rigid piston along the global +z axis; per-face velocity "
                    "scaled by dot(n_hat, z_hat)"
                ),
            },
            "refused": {
                "arbitrary_piston_axis": "the axial piston axis is fixed to global +z",
                "taper": "not exposed",
                "annular": "not exposed",
                "per_face": "not exposed",
                "callable": "not exposed",
            },
        },
        "tag_identity": {
            "field": "velocity_sources",
            "shape": "{physical_tag: amplitude}",
            "reported_back": "SolveConfig.source_tag",
        },
    }


def _observation() -> dict[str, Any]:
    return {
        "polar_cuts": {
            "field": "observation.planes",
            "supported_planes": ["horizontal", "vertical", "diagonal"],
            "horizontal_plane": "x-z",
            "vertical_plane": "y-z",
            "diagonal": "horizontal rotated toward vertical by inclination_deg",
            "angle_range_deg": [-180.0, 180.0],
            "must_include_zero_deg": True,
            "centre": "the solver coordinate origin",
        },
        "sphere_grid": {
            "field": "observation.sphere_grid",
            "supported": True,
            "layout": "theta-major, theta in [0, sphere_theta_max_deg], phi in [0, 360)",
            "min_theta_count": 2,
            "min_phi_count": 3,
            "max_points": 100_000,
        },
        "frame": {
            "field": "frame_override",
            "supported": "rigid translation only",
            "required_orientation": {"axis": "+z", "u": "+x", "v": "+y"},
            "orientation_tolerance": 1.0e-3,
            "refused": {
                "tilted_frame": (
                    "the vendored solver computes its cuts in the mesh's own "
                    "global frame, so a rotated frame would silently measure "
                    "the wrong arc"
                ),
                "frame_inference": (
                    "this package reads a Gmsh 2.2 file for tag areas, not "
                    "for a horn's axis; it cannot infer a frame"
                ),
            },
        },
        "named_origin": {
            "field": "observation.origin",
            "supported_values": [None, "mouth", "throat"],
            "requires_frame_override": True,
            "requires_frame_note": (
                "a named origin without frame_override is refused; it used to "
                "be accepted and then ignored, producing identical requests"
            ),
        },
        "arbitrary_points": {
            "supported": False,
            "reason": "the request carries a distance and an angle grid, not points",
        },
        "post_solve_field_replay": {
            "supported": False,
            "reason": (
                "surface traces can be retained, but this package exports no "
                "exterior evaluator that consumes them"
            ),
        },
    }


def _returned_quantities() -> dict[str, Any]:
    return {
        "pressure_complex": {
            "shape": "(frequencies, planes, angles)",
            "unit": "Pa per unit normal acceleration",
        },
        "spl_db": {
            "quantity": "absolute SPL",
            "unit": "dB re 20 uPa",
        },
        "directivity_db": {
            "quantity": "on-axis-normalized directivity",
            "unit": "dB",
            "reference": "first sample of smallest |angle|, per frequency and plane",
            "alias": "spl_norm_db",
            "floor_note": (
                "amplitudes are floored at -120 dB re 20 uPa before the "
                "subtraction, so a null reference cannot produce inf - inf"
            ),
        },
        "impedance": {
            "quantity": "area-weighted mean pressure on the source tag",
            "unit": "Pa per unit normal acceleration",
            "normalized_to_rho_c": False,
            "legacy_name": True,
            "legacy_note": (
                "named 'impedance' for cross-package compatibility; it is a "
                "raw mean pressure, not a mechanical or acoustic impedance"
            ),
        },
        "surface_traces": {
            "field": "surface_traces",
            "supported": True,
            "default": False,
            "pressure": "P1, one complex value per vertex",
            "neumann": "DP0, one complex value per face",
            "domain_note": (
                "under a reduced-mesh symmetry solve these cover the "
                "fundamental domain, not the full body"
            ),
        },
        "sphere_pressure_complex": {
            "supported": True,
            "populated_when": "observation.sphere_grid is set",
        },
        "acoustic_power": {
            "supported": False,
            "reason": "no surface or sphere power integration is exposed",
        },
        "per_tag_mean_pressures": {
            "supported": False,
            "reason": "one source tag, so there is one mean pressure",
        },
    }


def _quadrature(backend: str) -> dict[str, Any]:
    regular_modes = ["fixed", "wavelength"] if backend == BEAT_CPU else ["fixed"]
    entry: dict[str, Any] = {
        "regular": {
            "field": "quadrature_order",
            "supported_orders": list(SUPPORTED_QUADRATURE_ORDERS),
            "gauss_points_for_order": {"1": 1, "2": 3, "4": 6},
            "refused_orders_note": (
                "the vendored triangle rule has exactly these three rules and "
                "returns the 3-point rule for every other value, so an order "
                "of 6 would be less accurate than the default while reading "
                "like a refinement; SolveConfig refuses anything outside "
                "supported_orders"
            ),
            "default_order": 4,
            "mode_field": "regular_quadrature_mode",
            "supported_modes": regular_modes,
            "default_mode": "wavelength" if backend == BEAT_CPU else "fixed",
        },
        "singular": {
            "field": "singular_order",
            "scheme": "Duffy, 1-D Gauss order for coincident and adjacent pairs",
            "range": [1, 12],
            "default": 4,
            "above_4_requires_precision": "double",
        },
        "near_pair": {
            "field": "near_correction",
            "default": False,
            "cutoff_field": "near_correction_cutoff",
            "order_field": "near_correction_order",
            "min_order": 4,
        },
    }
    if backend != BEAT_CPU:
        entry["regular"]["refused_modes"] = {
            "wavelength": (
                "wavelength-driven regular quadrature is implemented for the "
                "BEAT CPU backend only; the Julia driver refuses it at solve "
                "time rather than SolveConfig refusing it at construction"
            )
        }
    refusal = _NEAR_CORRECTION_REFUSALS.get(backend)
    if refusal is None:
        entry["near_pair"]["supported"] = True
        # Properties of the correction, so they are stated only where the
        # correction exists. Left unconditional they read as claims about a
        # feature the same entry refuses two lines further down.
        entry["near_pair"]["covers_image_pairs"] = True
        entry["near_pair"]["off_is_bitwise_identical"] = True
    else:
        entry["near_pair"]["supported"] = False
        entry["near_pair"]["reason"] = refusal
    return entry


def _frequencies() -> dict[str, Any]:
    """What a caller may ask to have solved, and in what order it comes back.

    The order clause is the load-bearing one. ``solve_frequencies`` solves the
    list in the order given -- a caller may hand over a live-plotting order --
    and every returned array follows that order rather than being sorted, so a
    consumer that assumes an ascending axis reads the right numbers against
    the wrong frequencies. That is invisible in a level plot and obvious in a
    phase one, which is exactly the class of defect this report exists for.
    """

    defaults = SolveConfig()
    return {
        "explicit_list": {
            "entry_point": "solve_frequencies(mesh_path, frequencies_hz, config)",
            "field": "frequencies_hz",
            "unit": "Hz",
            "minimum_count": 1,
            "order_preserved": True,
            "order_note": (
                "solved in the order given, and every returned array follows "
                "that order; a caller wanting the canonical ascending axis "
                "sorts the result afterwards"
            ),
            "refused": {
                "empty": "an empty frequency list is refused",
                "non_finite": "NaN and infinite frequencies are refused",
                "non_positive": "zero and negative frequencies are refused",
                "duplicate": (
                    "a repeated frequency is refused; the result axis is the "
                    "request, so a duplicate would return two rows nothing "
                    "distinguishes"
                ),
            },
            "validated_before": (
                "any Julia resolution, worker start or mesh read, so a bad "
                "list costs nothing"
            ),
        },
        "generated_grid": {
            "entry_point": "solve(mesh_path, config)",
            "fields": ["freq_min_hz", "freq_max_hz", "freq_count", "freq_spacing"],
            "spacings": ["log", "linear"],
            "default_spacing": defaults.freq_spacing,
            "default_count": defaults.freq_count,
            "default_range_hz": [defaults.freq_min_hz, defaults.freq_max_hz],
            "endpoints": "inclusive",
            "implementation": "numpy geomspace for 'log', linspace for 'linear'",
        },
        "solved_precision_note": (
            "the driver casts the list to Float32 before solving; see the "
            "precision entry's solved_frequency_note"
        ),
    }


def _completion() -> dict[str, Any]:
    return {
        "statuses": {
            "completed": {
                "meaning": "every requested frequency was returned",
                "result_flags": {"cancelled": False, "is_partial": False},
                "invariant": (
                    "a 'completed' event carrying fewer results than "
                    "requested raises instead of returning"
                ),
            },
            "cancelled": {
                "meaning": (
                    "the caller stopped the sweep (on_frequency_result "
                    "returning exactly False, or a cancel written to the job "
                    "directory)"
                ),
                "result_flags": {"cancelled": True, "is_partial": "len < requested"},
                "note": (
                    "cancellation is the only way a short result is returned; "
                    "a sweep cancelled after the last frequency is complete "
                    "and still marked cancelled"
                ),
            },
            "failed": {
                "meaning": (
                    "EOF without a terminal event, a mismatched angle grid, "
                    "an out-of-order frequency echo, more results than "
                    "frequencies, or a raising callback"
                ),
                "delivery": "raises; no SolveResult is returned",
            },
        },
        "silent_truncation": {"possible": False},
        "cleanup": {
            "scope": "job directory and the persistent worker's turn",
            "guarantee": (
                "released in a finally block on every exit, including a "
                "callback raising mid-stream"
            ),
        },
        "streaming": {
            "progress_callback": "progress_callback(freq_index, total, frequency_hz)",
            "result_callback": "on_frequency_result(freq_index, frequency_hz, log_entry)",
        },
        "timings": {
            "field": "SolveResult.timings",
            "unit": "s",
            "always_present_keys": ["assembly_s", "solve_s", "field_s", "total_s"],
            "note": (
                "the first three are summed over the solved frequencies from "
                "the driver's own per-frequency report; total_s is this "
                "process's wall clock over the whole sweep and therefore "
                "includes worker start, transport and any cold compile"
            ),
        },
        "persistent_worker": {
            "supported": True,
            "default": True,
            "outlives_process": True,
            "identity_includes": [
                "worker protocol version",
                "julia executable",
                "solver script",
                "julia project",
                "julia sysimage",
                "thread count",
                "package version",
                "package content fingerprint",
                "keyed environment",
            ],
        },
    }


def _mesh_input() -> dict[str, Any]:
    return {
        "formats": ["gmsh 2.2 ascii"],
        "elements": ["triangle"],
        "read_for": ["per-tag areas", "vertex count", "triangle count"],
        "scale_field": "mesh_scale",
        "validation": {
            "topology_checks": False,
            "normal_repair": False,
            "reason": (
                "this package reads mesh facts; it does not compact, validate "
                "or repair a mesh the way a full preprocessing layer would"
            ),
        },
        "multiple_bodies": {
            "supported": False,
            "reason": (
                "the driver can ingest several meshes; the package request "
                "compiler supplies exactly one"
            ),
        },
    }


def _exterior_mode(backend: str) -> dict[str, Any]:
    return {
        "supported": True,
        "formulation": _formulation(backend),
        "precision": _precision(backend),
        "symmetry": _symmetry(),
        "ground_plane": _ground_plane(),
        "sources": _sources(),
        "frequencies": _frequencies(),
        "observation": _observation(),
        "quantities": _returned_quantities(),
        "quadrature": _quadrature(backend),
        "completion": _completion(),
        "mesh_input": _mesh_input(),
    }


_UNSUPPORTED_MODES = {
    "exterior_robin": (
        "no admittance or impedance boundary condition is exposed for the "
        "exterior problem; the coupled physical engine's impedance concepts "
        "are not the same feature"
    ),
    "infinite_baffle": (
        "no aperture-coupled path is implemented in this package; a rigid "
        "ground image is not equivalent to an infinite baffle"
    ),
    "circsym_m0": "no axisymmetric meridian solver is present",
    "coupled_fem_bem_lem": (
        "the vendored engine implements it and this package exposes no "
        "request compiler, preparation or result contract for it, so no "
        "caller can reach it through the supported API"
    ),
}


def _modes(backend: str) -> dict[str, Any]:
    modes: dict[str, Any] = {"exterior": _exterior_mode(backend)}
    for name in SOLVE_MODES:
        if name == "exterior":
            continue
        modes[name] = {"supported": False, "reason": _UNSUPPORTED_MODES[name]}
    return modes


def backend_capabilities(backend: str) -> dict[str, Any]:
    """The capability entry for one backend.

    Raises ``ValueError`` for a backend this package does not have, so a typo
    is a failure rather than an empty report.
    """

    if backend not in BEAT_BACKENDS:
        raise ValueError(
            "unknown BEAT backend "
            f"{backend!r}; expected one of {', '.join(BEAT_BACKENDS)}"
        )
    return {
        "backend": backend,
        "julia_project": _BACKEND_PROJECTS[backend],
        "device_class": "cpu" if backend == BEAT_CPU else "gpu",
        "user_facing": True,
        "user_facing_note": (
            "the CPU backend is a selectable engine wherever its runtime has "
            "been provisioned (provision --backend cpu instantiates the project "
            "and proves it with a 1 kHz solve); HORNLAB_BEAT_FORCE_CPU=1 only "
            "skips that evidence for CI and regression runs. Readiness is "
            "recorded per backend, so a host can have this and an accelerator "
            "provisioned at the same time"
        )
        if backend == BEAT_CPU
        else "",
        "bitwise_reproducible": backend == BEAT_CPU,
        "bitwise_reproducible_note": (
            "the accelerator assemblies accumulate singular corrections "
            "through atomics and are not bit-reproducible run to run"
        )
        if backend != BEAT_CPU
        else "the CPU reference mode is deterministic",
        "modes": _modes(backend),
    }


def capability_report(*, backend: str | None = None) -> dict[str, Any]:
    """The versioned capability contract of this package, per backend and mode.

    ``backend`` restricts the report to one backend; the default covers all
    four. The result is a plain JSON-serialisable dictionary built from
    constants, so two runs on two machines produce byte-identical output for
    the same commit. Nothing here probes hardware or a Julia runtime -- call
    :func:`hornlab_beat_bem.beat_engine_status` for that.
    """

    backends = BEAT_BACKENDS if backend is None else (backend,)
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "package": "hornlab-beat-bem",
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "engine": "BEAT Engine (boundary-lab), vendored",
        "availability_probe": "hornlab_beat_bem.beat_engine_status",
        "conventions": _conventions(),
        "solve_modes": list(SOLVE_MODES),
        "backends": {name: backend_capabilities(name) for name in backends},
    }
