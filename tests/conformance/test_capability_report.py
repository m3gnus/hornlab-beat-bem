"""The capability report is a contract, so it is tested like one."""

from __future__ import annotations

import json

import numpy as np
import pytest

import hornlab_beat_bem as beat
from hornlab_beat_bem.capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    SOLVE_MODES,
    backend_capabilities,
    capability_report,
)
from hornlab_beat_bem.config import (
    GROUND_PLANE_AXES,
    SUPPORTED_QUADRATURE_ORDERS,
    GroundPlane,
    ObservationConfig,
    SolveConfig,
)

#: The keys ``sweep._request_payload`` puts on the wire, pinned so that adding,
#: renaming or dropping one is a failing test rather than a silent change to a
#: contract the capability report gives a version number to. The version is an
#: integer and cannot notice a reshaped payload on its own -- that was the gap
#: this pair of lists closes.
EXPECTED_REQUEST_KEYS = ["config", "frequencies_hz", "schema_version"]

#: Every key a default config puts in the request's ``config`` block.
EXPECTED_REQUEST_CONFIG_KEYS = [
    "axial_offset",
    "distance",
    "flat_target_normalization_enabled",
    "freq_count",
    "freq_max",
    "freq_min",
    "max_angle",
    "mesh_file",
    "meshes",
    "min_angle",
    "quadrature_order",
    "rho",
    "scale_factor",
    "singular_order",
    "sound_speed",
    "source_motion",
    "spherical_sampling_enabled",
    "step_size",
    "symmetry",
    "tag_throat",
]

#: The keys the optional features add on top. Together with the list above,
#: this is the whole payload shape.
EXPECTED_OPTIONAL_REQUEST_CONFIG_KEYS = [
    "diagonal_enabled",
    "diagonal_inclination_deg",
    "ground_plane_min_clearance_m",
    "near_correction_cutoff",
    "near_correction_enabled",
    "near_correction_order",
    "regular_quadrature_mode",
    "solve_precision",
    "spherical_grid",
    "surface_traces_enabled",
]


def test_the_report_is_exported_from_the_package_namespace():
    assert beat.capability_report is capability_report
    assert beat.CAPABILITY_SCHEMA_VERSION == CAPABILITY_SCHEMA_VERSION


def test_the_report_carries_a_schema_version_and_the_request_version():
    report = capability_report()
    assert isinstance(report["schema_version"], int)
    assert report["schema_version"] == CAPABILITY_SCHEMA_VERSION
    assert report["request_schema_version"] == REQUEST_SCHEMA_VERSION


def test_the_request_schema_version_matches_the_request_the_package_builds():
    """A version that drifts from the payload is worse than none at all."""

    from hornlab_beat_bem.sweep import _request_payload

    payload = _request_payload(
        "mesh.msh", np.asarray([1000.0]), SolveConfig(), (0.0, 0.0, 0.0)
    )
    assert payload["schema_version"] == REQUEST_SCHEMA_VERSION


def test_the_request_payload_shape_is_pinned_not_only_its_version_number():
    """An integer cannot notice a renamed key, so pin the shape as well.

    ``request_schema_version`` is published in the capability report as the
    request contract a consumer is paired with. Comparing only the integer
    against the integer the payload carries proves they agree about a number
    and nothing about the payload, so a key added, renamed or dropped here
    would keep version 2 while changing what version 2 means. These lists are
    the deliberate decision: change the payload, change them, and decide in
    the same breath whether the version moves.
    """

    from hornlab_beat_bem.sweep import _request_payload

    frequencies = np.asarray([1000.0])
    default = _request_payload("mesh.msh", frequencies, SolveConfig(), (0.0, 0.0, 0.0))
    assert sorted(default) == EXPECTED_REQUEST_KEYS
    assert sorted(default["config"]) == EXPECTED_REQUEST_CONFIG_KEYS

    # Every conditional key at once, so the optional half of the shape is
    # pinned too rather than only the part a default config reaches.
    maximal_config = SolveConfig(
        solve_precision="double",
        near_correction=True,
        surface_traces=True,
        regular_quadrature_mode="fixed",
        ground_plane=GroundPlane(enabled=True, axis="y"),
        observation=ObservationConfig(
            planes=["horizontal", "vertical", "diagonal"], sphere_grid=(2, 3)
        ),
    )
    maximal = _request_payload("mesh.msh", frequencies, maximal_config, (0.0, 0.0, 0.0))
    assert sorted(maximal) == EXPECTED_REQUEST_KEYS
    assert sorted(maximal["config"]) == sorted(
        EXPECTED_REQUEST_CONFIG_KEYS + EXPECTED_OPTIONAL_REQUEST_CONFIG_KEYS
    )
    assert maximal["schema_version"] == REQUEST_SCHEMA_VERSION


def test_every_backend_and_every_solve_mode_has_an_entry():
    report = capability_report()
    assert set(report["backends"]) == set(beat.BEAT_BACKENDS)
    for backend, entry in report["backends"].items():
        assert entry["backend"] == backend
        assert set(entry["modes"]) == set(SOLVE_MODES)


def test_the_report_is_json_serialisable_and_deterministic():
    """Two calls on the same commit must produce identical bytes."""

    first = json.dumps(capability_report(), sort_keys=True)
    second = json.dumps(capability_report(), sort_keys=True)
    assert first == second


def test_the_report_carries_no_host_specific_paths():
    """It describes the package, not the machine -- so it must not leak one.

    ``beat_engine_status`` is the host probe; keeping absolute paths out of
    here is what makes two machines' reports comparable, and it is also what
    keeps a committed or published report free of local paths.
    """

    serialised = json.dumps(capability_report())
    for fragment in ("/Users/", "/home/", "C:\\Users", "/private/tmp", "/var/folders"):
        assert fragment not in serialised
    for entry in capability_report()["backends"].values():
        assert entry["julia_project"].startswith("hornlab_beat_bem/")


def test_a_backend_filter_narrows_the_report():
    report = capability_report(backend=beat.BEAT_METAL)
    assert set(report["backends"]) == {beat.BEAT_METAL}


def test_an_unknown_backend_is_a_failure_rather_than_an_empty_report():
    with pytest.raises(ValueError, match="unknown BEAT backend"):
        backend_capabilities("opencl")


def test_only_the_exterior_mode_is_reachable_and_the_rest_say_why():
    for backend in beat.BEAT_BACKENDS:
        modes = backend_capabilities(backend)["modes"]
        assert modes["exterior"]["supported"] is True
        for name in ("exterior_robin", "infinite_baffle", "circsym_m0", "coupled_fem_bem_lem"):
            assert modes[name]["supported"] is False
            assert modes[name]["reason"]


def test_the_coupled_mode_is_reported_absent_because_the_package_cannot_reach_it():
    """The engine implements it; the package exposes no way to ask for it."""

    entry = backend_capabilities(beat.BEAT_CPU)["modes"]["coupled_fem_bem_lem"]
    assert "request compiler" in entry["reason"]


def test_the_exterior_formulation_is_burton_miller_and_not_selectable():
    exterior = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]
    formulation = exterior["formulation"]
    assert formulation["name"] == "burton_miller"
    assert formulation["selectable"] is False
    assert set(formulation["refused"]) >= {"chief", "complex_k"}


def test_double_precision_is_reported_on_the_cpu_backend_only():
    for backend in beat.BEAT_BACKENDS:
        precision = backend_capabilities(backend)["modes"]["exterior"]["precision"]
        requested = precision["requested_values"]
        if backend == beat.BEAT_CPU:
            assert requested == ["single", "double"]
            SolveConfig(beat_backend=backend, solve_precision="double")
        else:
            assert requested == ["single"]
            with pytest.raises(NotImplementedError):
                SolveConfig(beat_backend=backend, solve_precision="double")


def test_the_float32_result_serialisation_limit_is_reported_on_every_backend():
    """A 'double' solve is Float64 inside and Float32 on the wire; say so."""

    for backend in beat.BEAT_BACKENDS:
        precision = backend_capabilities(backend)["modes"]["exterior"]["precision"]
        assert precision["result_serialization"] == "float32"
        assert precision["solved_frequency_precision"] == "float32"
        assert precision["actual_assembly_and_solve"]["single"] == "float32"


def test_the_reported_ground_axes_are_the_axes_the_package_advertises():
    ground = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["ground_plane"]
    assert tuple(ground["supported_axes"]) == GROUND_PLANE_AXES
    assert set(ground["refused_axes"]) == {"x", "z"}
    assert ground["physical_radiator_count"] == 1


def test_the_reported_symmetry_modes_match_the_mapping_the_solver_is_given():
    from hornlab_beat_bem.config import beat_symmetry_mode

    symmetry = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["symmetry"]
    for entry in symmetry["supported"].values():
        assert beat_symmetry_mode(entry["value"]) == entry["solver_image_mode"]
    assert set(symmetry["refused"]) == {"xz", "xy"}


def test_near_correction_is_reported_on_the_cpu_backend_only():
    """Every accelerator refuses it, and Metal is the one that used to lie.

    Metal was reported supported and accepted by ``SolveConfig`` while the
    correction never ran: ``BeatEngineCore``'s ``:metal`` branch forwards no
    near-correction cache and the Metal assembly takes none, so the flag
    changed the status line and not the answer -- measured at 1.8e-7 relative,
    which is that backend's own run-to-run noise, against 1.4e-4 on the CPU.
    Pinning the whole set here rather than only the CPU entry is deliberate:
    the report reads the refusal table ``SolveConfig`` enforces, so the two
    cannot disagree, and this test is what makes removing a backend from that
    table a decision rather than an edit.
    """

    supported = {
        backend: backend_capabilities(backend)["modes"]["exterior"]["quadrature"][
            "near_pair"
        ]["supported"]
        for backend in beat.BEAT_BACKENDS
    }
    assert supported == {
        beat.BEAT_CPU: True,
        beat.BEAT_METAL: False,
        beat.BEAT_CUDA: False,
        beat.BEAT_ROCM: False,
    }
    for backend in (beat.BEAT_METAL, beat.BEAT_CUDA, beat.BEAT_ROCM):
        entry = backend_capabilities(backend)["modes"]["exterior"]["quadrature"][
            "near_pair"
        ]
        assert entry["reason"]
        # Properties of a correction that is not there are not facts about
        # this backend. They used to be reported True beside supported False.
        assert "covers_image_pairs" not in entry
        assert "off_is_bitwise_identical" not in entry
        with pytest.raises(NotImplementedError, match="near_correction"):
            SolveConfig(beat_backend=backend, near_correction=True)
    cpu = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["quadrature"][
        "near_pair"
    ]
    assert cpu["covers_image_pairs"] is True
    assert cpu["off_is_bitwise_identical"] is True


def test_the_reported_quadrature_orders_are_the_orders_the_engine_distinguishes():
    """1, 2 and 4 are three rules; 6 is the order-2 rule wearing a bigger number."""

    regular = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["quadrature"][
        "regular"
    ]
    assert regular["supported_orders"] == list(SUPPORTED_QUADRATURE_ORDERS)
    assert set(regular["gauss_points_for_order"]) == {
        str(order) for order in SUPPORTED_QUADRATURE_ORDERS
    }
    assert regular["default_order"] in SUPPORTED_QUADRATURE_ORDERS
    for order in SUPPORTED_QUADRATURE_ORDERS:
        SolveConfig(quadrature_order=order)
    for order in (0, 3, 5, 6, 8):
        with pytest.raises(ValueError, match="quadrature_order"):
            SolveConfig(quadrature_order=order)


def test_the_frequency_contract_states_the_order_rule_and_the_generated_grid():
    """The order clause is the one a consumer can get wrong silently."""

    frequencies = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["frequencies"]
    explicit = frequencies["explicit_list"]
    assert explicit["order_preserved"] is True
    assert set(explicit["refused"]) == {
        "empty",
        "non_finite",
        "non_positive",
        "duplicate",
    }
    grid = frequencies["generated_grid"]
    defaults = SolveConfig()
    assert grid["default_spacing"] == defaults.freq_spacing
    assert grid["default_count"] == defaults.freq_count
    assert grid["default_range_hz"] == [defaults.freq_min_hz, defaults.freq_max_hz]
    assert set(grid["spacings"]) == {"log", "linear"}


def test_the_generated_grid_is_the_grid_the_report_describes():
    from hornlab_beat_bem.sweep import generated_frequency_grid

    grid = generated_frequency_grid(SolveConfig())
    described = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["frequencies"][
        "generated_grid"
    ]
    assert grid.size == described["default_count"]
    assert [grid[0], grid[-1]] == pytest.approx(described["default_range_hz"])


def test_the_reported_timings_are_the_keys_a_result_actually_carries():
    """Named because a record's provenance quotes them; unnamed they rot."""

    completion = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["completion"]
    assert completion["timings"]["field"] == "SolveResult.timings"
    assert completion["timings"]["always_present_keys"] == [
        "assembly_s",
        "solve_s",
        "field_s",
        "total_s",
    ]


def test_wavelength_quadrature_is_reported_as_a_cpu_only_mode():
    for backend in beat.BEAT_BACKENDS:
        regular = backend_capabilities(backend)["modes"]["exterior"]["quadrature"]["regular"]
        if backend == beat.BEAT_CPU:
            assert regular["supported_modes"] == ["fixed", "wavelength"]
            assert regular["default_mode"] == "wavelength"
        else:
            assert regular["supported_modes"] == ["fixed"]
            assert "wavelength" in regular["refused_modes"]


def test_the_source_contract_reports_one_unit_amplitude_tag():
    sources = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["sources"]
    assert sources["count"]["supported"] == 1
    assert sources["amplitude"]["supported"] == [1.0]
    assert set(sources["profiles"]["supported"]) == {"normal", "axial"}
    assert set(sources["profiles"]["refused"]) >= {"taper", "annular", "per_face", "callable"}


def test_the_completion_statuses_cover_what_a_sweep_can_end_as():
    completion = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["completion"]
    assert set(completion["statuses"]) == {"completed", "cancelled", "failed"}
    assert completion["silent_truncation"]["possible"] is False
    assert completion["statuses"]["failed"]["delivery"].startswith("raises")


def test_the_reported_worker_identity_covers_every_field_of_the_real_key():
    """A field added to ``worker_key`` must be described here, or not at all.

    The list is prose and the key is code, so no assertion can prove they say
    the same thing. What this does prove is that they are the same *size*: a
    ninth adoption criterion cannot appear in the worker registry while the
    published contract still names eight.
    """

    from pathlib import Path

    from hornlab_beat_bem.worker_registry import worker_key

    key = worker_key(
        julia_executable="julia",
        solver_script=Path("solver.jl"),
        julia_project=None,
        julia_sysimage=None,
        julia_threads="auto",
        package_version="0.1.0",
    )
    reported = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["completion"][
        "persistent_worker"
    ]["identity_includes"]
    assert len(reported) == len(key), (
        f"the report names {len(reported)} adoption criteria and worker_key "
        f"carries {len(key)}: {sorted(key)}"
    )


def test_the_conventions_block_states_the_phase_and_drive_contract():
    conventions = capability_report()["conventions"]
    assert conventions["time_convention"] == "exp(-i*omega*t)"
    assert conventions["outgoing_wave"] == "exp(+i*k*r)"
    assert conventions["drive_quantity"] == "normal acceleration"
    assert conventions["length_unit"] == "m"
    assert conventions["medium_defaults"]["sound_speed_m_s"] == SolveConfig().sound_speed
    assert conventions["medium_defaults"]["air_density_kg_m3"] == SolveConfig().air_density


def test_arbitrary_observation_points_and_field_replay_are_reported_absent():
    observation = backend_capabilities(beat.BEAT_CPU)["modes"]["exterior"]["observation"]
    assert observation["arbitrary_points"]["supported"] is False
    assert observation["post_solve_field_replay"]["supported"] is False
    assert observation["named_origin"]["requires_frame_override"] is True
