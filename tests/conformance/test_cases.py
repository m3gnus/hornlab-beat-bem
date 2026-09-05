"""The conformance cases, run against a real Julia worker (slow).

Each case is run once, by a module-scoped fixture, and the assertions read the
record it produced. Running a case per assertion would pay for a BEM solve to
check a dictionary key.

The comparison against hornlab-metal-bem is asserted only when it ran. When
that package is absent -- which is every CI runner -- the record carries the
reason and these tests still assert the record's shape, so an absent
counterpart is a recorded fact rather than a pytest skip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import hornlab_beat_bem as beat
from hornlab_beat_bem._constants import REFERENCE_PRESSURE

from .analytic import (
    icosphere,
    icosphere_surface_area,
    level_and_phase_error,
    pulsating_sphere_pressure_per_acceleration,
)
from .cases import (
    COMPARISON_LEVEL_TOLERANCE_DB,
    COMPARISON_PHASE_TOLERANCE_DEG,
    CONTROL_MIN_PHASE_ERROR_DEG,
    SPHERE_ANGLE_COUNT,
    SPHERE_FREQUENCY_HZ,
    SPHERE_LEVEL_TOLERANCE_DB,
    SPHERE_OBSERVATION_M,
    SPHERE_PHASE_TOLERANCE_DEG,
    SPHERE_RADIUS_M,
    SPHERE_SUBDIVISIONS,
    _accept_analytic_sphere,
    _compare_sphere,
    case_by_name,
    sphere_reference_pressure,
)
from .harness import run_case

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def julia():
    executable = beat.discover_julia()
    if executable is None:
        pytest.skip("no Julia executable (set HORNLAB_BEAT_JULIA)")
    return executable


@pytest.fixture(scope="module")
def records(julia, tmp_path_factory):
    """Run every solve case once and hand back the records by name."""

    root = tmp_path_factory.mktemp("conformance")
    produced = {}
    try:
        for name in (
            "exterior_contract_two_frequencies",
            "analytic_pulsating_sphere_phase",
        ):
            produced[name] = run_case(
                case_by_name(name),
                work_dir=root / "work",
                output_dir=root / "results",
                julia_executable=julia,
            )
        yield produced
    finally:
        beat.shutdown_workers()


# ---------------------------------------------------------------------------
# The reference and the mesh it is exact on, checked before any solve is blamed
# ---------------------------------------------------------------------------


def test_the_icosphere_generator_is_a_closed_outward_oriented_sphere():
    vertices, faces = icosphere(SPHERE_RADIUS_M, SPHERE_SUBDIVISIONS)
    assert faces.shape == (20 * 4**SPHERE_SUBDIVISIONS, 3)
    assert np.allclose(np.linalg.norm(vertices, axis=1), SPHERE_RADIUS_M)
    v1, v2, v3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    centroids = (v1 + v2 + v3) / 3.0
    assert np.all(np.sum(np.cross(v2 - v1, v3 - v1) * centroids, axis=1) > 0.0)
    directed = set()
    for a, b, c in faces:
        directed.update({(a, b), (b, c), (c, a)})
    assert all((b, a) in directed for a, b in directed), "the surface is not closed"
    exact = 4.0 * math.pi * SPHERE_RADIUS_M**2
    deficit = icosphere_surface_area(SPHERE_RADIUS_M, SPHERE_SUBDIVISIONS) / exact - 1.0
    assert -0.01 < deficit < 0.0, "a faceted sphere is inscribed, so its area is short"


def test_the_closed_form_has_the_expected_low_frequency_limit_and_phase_advance():
    incompressible = pulsating_sphere_pressure_per_acceleration(
        SPHERE_OBSERVATION_M,
        SPHERE_RADIUS_M,
        1.0e-6,
        air_density=1.2041,
        sound_speed=343.0,
    )
    assert incompressible.real == pytest.approx(
        1.2041 * SPHERE_RADIUS_M**2 / SPHERE_OBSERVATION_M, rel=1e-6
    )
    assert incompressible.imag == pytest.approx(0.0, abs=1e-9)

    # An outgoing wave under exp(-i*omega*t) advances its phase with distance;
    # under the opposite convention it would retard. This is the property the
    # solve case is scored on, asserted on the reference itself first.
    reference = sphere_reference_pressure()
    further = pulsating_sphere_pressure_per_acceleration(
        SPHERE_OBSERVATION_M + 0.01,
        SPHERE_RADIUS_M,
        SPHERE_FREQUENCY_HZ,
        air_density=1.2041,
        sound_speed=343.0,
    )
    k = 2.0 * math.pi * SPHERE_FREQUENCY_HZ / 343.0
    assert float(np.angle(further / reference)) == pytest.approx(k * 0.01, abs=1e-9)


def test_a_conjugated_field_is_invisible_to_a_level_gate_and_obvious_to_a_phase_one():
    """Why the analytic case asserts phase and cannot assert level instead."""

    reference = sphere_reference_pressure()
    conjugated = level_and_phase_error(np.conjugate(np.asarray([reference])), reference)
    assert conjugated["worst_level_error_db"] == pytest.approx(0.0, abs=1e-12)
    assert conjugated["worst_phase_error_deg"] > CONTROL_MIN_PHASE_ERROR_DEG


def _synthetic_sphere_result(field: np.ndarray) -> beat.SolveResult:
    """A ``SolveResult`` carrying an arbitrary field on the sphere case's grid."""

    pressure = np.asarray(field, dtype=np.complex128)
    with np.errstate(divide="ignore"):
        spl_db = 20.0 * np.log10(np.abs(pressure) / REFERENCE_PRESSURE)
    return beat.SolveResult(
        frequencies_hz=np.array([SPHERE_FREQUENCY_HZ]),
        pressure_complex=pressure,
        spl_db=spl_db,
        impedance=np.ones(1, dtype=np.complex128),
        observation_angles_deg=np.linspace(0.0, 180.0, SPHERE_ANGLE_COUNT),
        observation_planes=["horizontal", "vertical"],
        config=beat.SolveConfig(),
        mesh_info=beat.MeshInfo(0, 0, {}),
    )


def test_the_acceptance_check_rejects_a_conjugated_or_sign_flipped_measurement():
    """The gate bites on the *measurement*, not only on a shifted reference.

    ``_accept_analytic_sphere`` scores one measured field against the closed
    form and against two lies. That proves the lies are far from the reference;
    it does not by itself prove the acceptance would reject a solve that came
    back conjugated -- which is the defect this case exists to catch, and the
    one a real solve cannot be asked to produce on demand. So feed the
    acceptance a field that is exactly the closed form, then the same field
    conjugated and sign-flipped, and require the first to pass and the other
    two to fail on phase. The conjugated field's level error stays identically
    zero, which is the whole argument for scoring phase at all.
    """

    exact = np.full(
        (1, 2, SPHERE_ANGLE_COUNT), sphere_reference_pressure(), dtype=np.complex128
    )
    passing = _accept_analytic_sphere(_synthetic_sphere_result(exact))
    assert passing["against_closed_form"]["worst_phase_error_deg"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert passing["against_closed_form"]["worst_level_error_db"] == pytest.approx(
        0.0, abs=1e-9
    )

    for label, lie in (("conjugated", np.conjugate(exact)), ("sign_flipped", -exact)):
        with pytest.raises(AssertionError, match="phase error") as raised:
            _accept_analytic_sphere(_synthetic_sphere_result(lie))
        assert "level error" not in str(raised.value), (
            f"the {label} field was caught on level, so this run does not show "
            "that the phase assertion is the one doing the work"
        )


def test_a_band_violation_still_records_the_metrics_it_is_argued_from():
    """The scored table has to reach the record *before* the bands are judged.

    A cross-package disagreement is the one comparison outcome somebody will
    have to argue about, and the argument is the numbers. ``_compare_sphere``
    used to build them in a local and return them for the harness to assign,
    so a violation raised on the way out and left a comparison entry that said
    ``ran: true`` and carried no ``metrics`` at all -- the record could say the
    packages had been compared and not by how much.

    So force a violation: score a counterpart that is a flat 3 dB loud against
    a 0.5 dB band, and require the entry to hold both packages' scores anyway.
    """

    exact = np.full(
        (1, 2, SPHERE_ANGLE_COUNT), sphere_reference_pressure(), dtype=np.complex128
    )
    offset_db = 3.0
    comparison = {"package": "hornlab-metal-bem", "ran": True, "skip_reason": None}
    with pytest.raises(AssertionError, match="exceeds the .* dB band") as raised:
        _compare_sphere(
            _synthetic_sphere_result(exact),
            _synthetic_sphere_result(exact * 10.0 ** (offset_db / 20.0)),
            comparison,
        )
    assert "metal_bem_vs_closed_form" in str(raised.value)

    metrics = comparison["metrics"]
    assert metrics["beat_vs_closed_form"]["worst_level_error_db"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert metrics["metal_bem_vs_closed_form"][
        "worst_level_error_db"
    ] == pytest.approx(offset_db, abs=1e-9)
    assert metrics["beat_vs_metal_bem"]["worst_level_error_db"] == pytest.approx(
        offset_db, abs=1e-9
    )
    assert metrics["agreement_band"]["level_db"] == COMPARISON_LEVEL_TOLERANCE_DB


# ---------------------------------------------------------------------------
# The solve cases
# ---------------------------------------------------------------------------


def test_the_exterior_contract_case_passes_and_records_what_ran(records, julia):
    record = records["exterior_contract_two_frequencies"]
    assert record["status"] == "passed"
    assert record["solver"]["completion_status"] == "completed"
    assert record["solver"]["cancelled"] is False
    assert record["solver"]["is_partial"] is False
    assert record["solver"]["solved_frequencies_hz"] == [300.0, 500.0]
    assert record["case"]["frequencies_hz"] == [300.0, 500.0]
    assert record["mesh"]["sha256"] and record["mesh"]["n_triangles"] == 4
    assert record["request"]["solver_image_mode"] == "off"
    assert record["request"]["source_count"] == 1
    assert record["provenance"]["julia"]["executable"] == julia
    assert record["provenance"]["julia"]["version"]
    assert record["provenance"]["package"]["fingerprint"]
    assert record["worker_state_after"]["state"] == "warm"
    assert record["solve_wall_seconds"] > 0.0
    assert record["comparison"]["package"] == "hornlab-metal-bem"

    # The report names the timing keys; this is the only place they can be
    # checked against a result that a solver actually produced.
    from hornlab_beat_bem.capabilities import backend_capabilities

    reported = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["completion"][
        "timings"
    ]["always_present_keys"]
    assert set(record["solver"]["timings_s"]) == set(reported)


def test_the_solved_frequency_axis_follows_the_request_rather_than_sorting_it(
    julia, tmp_path
):
    """The report declares the order preserved, so prove it on a real solve.

    A consumer that assumes an ascending axis reads the right levels against
    the wrong frequencies, which a magnitude plot hides. The exterior case
    happens to request an ascending list, so it cannot show this; a descending
    one can.
    """

    from .cases import _tetrahedron_config, _tetrahedron_mesh

    mesh = _tetrahedron_mesh(tmp_path)
    result = beat.solve_frequencies(mesh, [500.0, 300.0], _tetrahedron_config(julia))
    assert [float(value) for value in result.frequencies_hz] == [500.0, 300.0]


def test_the_analytic_sphere_matches_the_closed_form_in_level_and_phase(records):
    measured = records["analytic_pulsating_sphere_phase"]["metrics"][
        "against_closed_form"
    ]
    assert measured["worst_level_error_db"] < SPHERE_LEVEL_TOLERANCE_DB
    assert measured["worst_phase_error_deg"] < SPHERE_PHASE_TOLERANCE_DEG


def test_the_analytic_sphere_case_rejects_a_conjugated_or_sign_flipped_field(records):
    """The convention gate. Both controls must miss by a wide margin.

    ``run_case`` already raises if these fail -- the acceptance function
    asserts them. What is added here is reading them back out of the record,
    so the evidence that this case can tell a conjugation from a correct
    answer is a number in a file rather than a claim in a docstring.
    """

    metrics = records["analytic_pulsating_sphere_phase"]["metrics"]
    for control in ("control_conjugated", "control_sign_flipped"):
        assert metrics[control]["worst_phase_error_deg"] > CONTROL_MIN_PHASE_ERROR_DEG
    assert metrics["control_conjugated"]["worst_level_error_db"] == pytest.approx(
        metrics["against_closed_form"]["worst_level_error_db"], abs=1e-12
    )


def test_the_analytic_sphere_record_pins_the_mesh_and_the_precision(records):
    record = records["analytic_pulsating_sphere_phase"]
    assert record["request"]["solve_precision"] == "double"
    assert record["mesh"]["n_triangles"] == 20 * 4**SPHERE_SUBDIVISIONS
    assert record["case"]["frequencies_hz"] == [SPHERE_FREQUENCY_HZ]
    assert record["metrics"]["isotropy_spread_db"] < SPHERE_LEVEL_TOLERANCE_DB


def test_the_metal_bem_comparison_either_ran_or_recorded_why_not(records):
    comparison = records["analytic_pulsating_sphere_phase"]["comparison"]
    assert comparison["package"] == "hornlab-metal-bem"
    if not comparison["ran"]:
        assert comparison["skip_reason"]
        return
    metrics = comparison["metrics"]
    assert (
        metrics["metal_bem_vs_closed_form"]["worst_level_error_db"]
        < COMPARISON_LEVEL_TOLERANCE_DB
    )
    assert (
        metrics["metal_bem_vs_closed_form"]["worst_phase_error_deg"]
        < COMPARISON_PHASE_TOLERANCE_DEG
    )
    if "beat_vs_metal_bem" in metrics:
        cross = metrics["beat_vs_metal_bem"]
        assert cross["worst_level_error_db"] < COMPARISON_LEVEL_TOLERANCE_DB
        assert cross["worst_phase_error_deg"] < COMPARISON_PHASE_TOLERANCE_DEG
