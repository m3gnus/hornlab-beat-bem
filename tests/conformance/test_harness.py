"""The harness itself: does a record say what actually ran?

These run no solver. They exist because a conformance record is only worth
anything if its provenance is real, and provenance is exactly the part that
fails silently -- a missing git SHA or an empty environment block looks like a
clean run.
"""

from __future__ import annotations

import json

import pytest

from .cases import all_cases, case_by_name
from .harness import (
    CONFORMANCE_RECORD_SCHEMA_VERSION,
    ConformanceCase,
    provenance,
    run_case,
    worker_state,
)


def test_every_case_has_a_unique_name_and_a_description():
    cases = all_cases()
    names = [case.name for case in cases]
    assert len(names) == len(set(names))
    for case in cases:
        assert case.description
        assert case.needs_solver == (case.make_config is not None)


def test_provenance_names_the_code_the_interpreter_and_the_host():
    facts = provenance(julia_executable=None)
    assert facts["package"]["path"]
    assert facts["package"]["fingerprint"]
    assert facts["package"]["git"]["sha"] or facts["package"]["git"]["source"]
    assert facts["python"]["executable"]
    assert facts["python"]["prefix"]
    assert facts["record_schema_version"] == CONFORMANCE_RECORD_SCHEMA_VERSION
    assert isinstance(facts["environment_overrides"], dict)


def test_worker_state_reports_cold_or_warm_with_its_evidence():
    state = worker_state()
    assert state["state"] in {"cold", "warm"}
    assert state["live_registry_hosts"] >= 0
    assert state["in_process_workers"] >= 0


def _assert_refusal_counts_add_up(metrics: dict) -> None:
    """Declared = exercised + excused, in one unit, with the lists to match.

    The counts went into a commit message as "102 declared, 98 exercised, 6
    excused" -- three numbers that cannot all be right, because two of them
    were path-instances and the third was distinct paths. Everything reported
    now is path-instances: one declared refusal on one backend.
    """

    assert "path-instances" in metrics["unit"]
    assert metrics["declared_count"] == len(metrics["refusals_declared"])
    assert metrics["refusal_count"] == len(metrics["refusals_verified"])
    assert metrics["excused_count"] == len(metrics["refusals_excused"])
    assert metrics["declared_count"] == metrics["refusal_count"] + (
        metrics["excused_count"]
    )
    # The unsupported non-exterior modes are counted separately: they are not
    # entries in the exterior refusal walk, and adding them to the exercised
    # side is how the total came out larger than the declarations.
    assert metrics["unsupported_mode_count"] == len(
        metrics["unsupported_modes_verified"]
    )
    assert set(metrics["refusals_declared"]) == set(metrics["refusals_verified"]) | set(
        metrics["refusals_excused"]
    )


def test_the_second_refusal_spelling_is_collected_and_exercised():
    """``supported: False`` + reason is a declared refusal, and is walked.

    The report spells a refusal two ways. The collector only knew the
    ``refused*`` keys, so the near-correction refusal was walked by a
    hand-written branch and recorded against ``quadrature.near_pair.refused``
    -- a path the report never declares. That is the transcription this case
    exists to remove, so both halves are pinned here: the API-absence entries
    are collected as declarations, and the one with a callable surface is
    exercised at the path the report actually uses.
    """

    from hornlab_beat_bem.capabilities import backend_capabilities

    from .cases import UNEXERCISABLE_REFUSALS, _check_declared_refusals
    from .cases import _declared_refusal_paths, _declared_unsupported_paths

    metal = backend_capabilities("metal")["modes"]["exterior"]
    unsupported = _declared_unsupported_paths(metal)
    assert "quadrature.near_pair" in unsupported
    assert {
        "sources.frequency_dependent_drive",
        "observation.arbitrary_points",
        "observation.post_solve_field_replay",
        "quantities.acoustic_power",
        "quantities.per_tag_mean_pressures",
        "mesh_input.multiple_bodies",
    } <= set(unsupported)
    assert set(unsupported) <= _declared_refusal_paths(metal)

    metrics = _check_declared_refusals()
    _assert_refusal_counts_add_up(metrics)
    assert "metal.quadrature.near_pair" in metrics["refusals_verified"]
    assert "metal.quadrature.near_pair.refused" not in metrics["refusals_declared"]
    # Every API-absence entry is excused rather than silently unexercised.
    for path in set(unsupported) - {"quadrature.near_pair"}:
        assert path in UNEXERCISABLE_REFUSALS
        assert f"metal.{path}" in metrics["refusals_excused"]


def test_a_static_case_writes_a_record_carrying_its_provenance(tmp_path):
    record = run_case(
        case_by_name("capability_refusals"),
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "results",
    )
    assert record["status"] == "passed"
    assert record["case"]["kind"] == "static"
    assert record["metrics"]["refusal_count"] > 0
    assert record["wall_seconds"] >= 0.0
    _assert_refusal_counts_add_up(record["metrics"])
    written = json.loads(
        (tmp_path / "results" / "capability_refusals.json").read_text(encoding="utf-8")
    )
    assert written["schema_version"] == CONFORMANCE_RECORD_SCHEMA_VERSION
    assert written["provenance"]["python"]["executable"]
    assert written["worker_state_before"]["state"] in {"cold", "warm"}


def test_a_case_without_a_counterpart_records_an_explicit_comparison_skip(tmp_path):
    """Never a pytest skip: a CI runner without metal-bem is a normal run."""

    record = run_case(
        case_by_name("capability_refusals"),
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "results",
    )
    comparison = record["comparison"]
    assert comparison["package"] == "hornlab-metal-bem"
    assert comparison["ran"] is False
    assert "no hornlab-metal-bem counterpart" in comparison["skip_reason"]


def test_a_comparison_that_cannot_run_records_why_rather_than_raising(tmp_path):
    """An unimportable or unusable counterpart is recorded, never inferred."""

    def explode():
        raise RuntimeError("no Metal device on this host")

    case = ConformanceCase(
        name="synthetic_comparison_probe",
        description="a case whose counterpart cannot be configured here",
        frequencies_hz=(1000.0,),
        make_metal_bem_config=explode,
        compare=lambda a, b, entry: None,
    )
    record = run_case(case, work_dir=tmp_path / "work", output_dir=tmp_path / "results")
    comparison = record["comparison"]
    assert comparison["ran"] is False
    assert comparison["skip_reason"]


def test_a_disagreeing_comparison_fails_the_case_and_keeps_its_record(
    tmp_path, monkeypatch
):
    """The comparison is part of the case, and its evidence survives it.

    ``case.compare`` used to run in the ``finally`` block, before the record
    was written: a comparison that raised -- which is what a comparison that
    disagrees does, now that the bands are asserted where they are recorded --
    destroyed the record and replaced the original failure with its own.

    The scored table has to survive too, and it did not: the scorer built its
    metrics in a local and returned them for the caller to assign, so a band
    assertion left a ``ran: true`` entry with no numbers in it -- the record
    said the packages were compared and could not say by how much. The scorer
    is handed the entry now and fills it in before asserting. Three halves are
    checked here: the assertion reaches the caller, the record on disk carries
    the case's own metrics and the comparison entry, and that entry carries
    the comparison metrics written before the disagreement was raised.
    """

    from . import harness

    entry = {
        "package": "hornlab-metal-bem",
        "ran": True,
        "skip_reason": None,
        "wall_seconds": 0.1,
    }
    monkeypatch.setattr(harness, "_metal_bem_run", lambda *a, **k: (entry, object()))

    def disagree(beat_result, metal_result, comparison):
        comparison["metrics"] = {"worst_level_error_db": 3.0}
        raise AssertionError("the two packages disagree by 3.0 dB")

    case = ConformanceCase(
        name="synthetic_comparison_disagreement",
        description="a case whose counterpart answers something else",
        static_check=lambda: {"checked": True},
        make_metal_bem_config=lambda: object(),
        compare=disagree,
    )
    with pytest.raises(AssertionError, match="disagree by 3.0 dB"):
        run_case(case, work_dir=tmp_path / "work", output_dir=tmp_path / "results")

    written = json.loads(
        (tmp_path / "results" / "synthetic_comparison_disagreement.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["status"] == "failed"
    assert "disagree by 3.0 dB" in written["error"]
    assert written["metrics"] == {"checked": True}
    assert written["comparison"]["ran"] is True
    assert written["comparison"]["metrics"]["worst_level_error_db"] == 3.0


def test_a_failing_case_is_recorded_before_it_is_raised(tmp_path):
    """Evidence must survive the failure it explains."""

    def fail() -> dict:
        raise AssertionError("the declared refusal did not fire")

    case = ConformanceCase(
        name="synthetic_failing_case",
        description="a case that fails its acceptance check",
        static_check=fail,
    )
    with pytest.raises(AssertionError, match="did not fire"):
        run_case(case, work_dir=tmp_path / "work", output_dir=tmp_path / "results")
    written = json.loads(
        (tmp_path / "results" / "synthetic_failing_case.json").read_text(encoding="utf-8")
    )
    assert written["status"] == "failed"
    assert "did not fire" in written["error"]


def test_a_solve_case_without_julia_is_recorded_as_a_failure(tmp_path, monkeypatch):
    """A missing runtime is a failed case, not a quietly successful one."""

    import hornlab_beat_bem as beat

    from . import harness

    monkeypatch.setattr(harness.beat, "discover_julia", lambda *args, **kwargs: None)
    with pytest.raises(beat.BeatUnavailableError):
        run_case(
            case_by_name("exterior_contract_two_frequencies"),
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "results",
        )
    written = json.loads(
        (tmp_path / "results" / "exterior_contract_two_frequencies.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["status"] == "failed"
    assert "Julia" in written["error"]
