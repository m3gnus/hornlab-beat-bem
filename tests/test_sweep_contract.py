"""The completion contract of ``solve_frequencies``, driven by a fake session.

These are fault-injection tests: they assert what the sweep does with a
protocol sequence, not that the Julia worker ever emits one. The gap they
cover is that a ``completed`` event used to end the loop unconditionally, so a
completion carrying no results returned a successful, empty ``SolveResult``
instead of failing -- and several exits left the session (and, with a
persistent worker, that worker's turn) to be released by garbage collection.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_beat_bem import ObservationConfig, SolveConfig
from hornlab_beat_bem import sweep
from hornlab_beat_bem.sweep import _WARMUP_TETRAHEDRON, solve_frequencies

_ANGLES = [0.0, 90.0]


def _result_event(frequency_hz: float, *, angle_count: int = 2) -> dict:
    row = [[1.0 / (index + 1) for index in range(angle_count)]]
    block = {"real": row, "imag": [[0.0] * angle_count]}
    return {
        "type": "result",
        "result": {
            "freq_hz": frequency_hz,
            "channel_names": ["main"],
            "horizontal_pressure": block,
            "vertical_pressure": block,
            "impedance": [[1.0, 0.0]],
            "timings": {"assembly_s": 0.1, "solve_s": 0.2, "field_s": 0.3},
        },
    }


class FakeSession:
    """A ``BeatSolveSession`` stand-in that never cleans up after itself.

    The real session releases its temporary job directory and the worker lock
    from the ``finally`` of ``events()``, which only runs once that generator
    is closed. Leaving that out here is the point: ``closed`` then counts only
    the sweep's *own* explicit lifetime handling, so a return to relying on
    garbage collection fails these tests instead of passing them by accident.
    """

    instances: list["FakeSession"] = []

    def __init__(self, events: list[dict], *, metadata: dict | None = None):
        self._events = events
        self.metadata = (
            {"polar_angle_deg": list(_ANGLES)} if metadata is None else metadata
        )
        self.closed = 0
        self.cancel_requests = 0

    def factory(self):
        def build(payload, **kwargs):
            FakeSession.instances.append(self)
            return self

        return build

    def events(self):
        yield from self._events

    def request_cancel(self) -> None:
        self.cancel_requests += 1

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def mesh_path(tmp_path):
    path = tmp_path / "tetrahedron.msh"
    path.write_text(_WARMUP_TETRAHEDRON, encoding="utf-8")
    return path


@pytest.fixture
def config():
    return SolveConfig(
        observation=ObservationConfig(
            planes=["horizontal", "vertical"],
            distance_m=1.0,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=2,
        )
    )


@pytest.fixture
def run(monkeypatch, mesh_path):
    """Run a sweep against a scripted event stream, with no Julia anywhere."""

    def _run(session: FakeSession, frequencies, config):
        monkeypatch.setattr(sweep, "discover_julia", lambda *a, **k: "/fake/julia")
        monkeypatch.setattr(sweep, "BeatSolveSession", session.factory())
        return solve_frequencies(mesh_path, frequencies, config)

    return _run


def test_complete_sweep_returns_every_requested_frequency(run, config):
    session = FakeSession(
        [_result_event(500.0), _result_event(1000.0), {"type": "completed"}]
    )
    result = run(session, [500.0, 1000.0], config)

    assert result.frequencies_hz.tolist() == [500.0, 1000.0]
    assert result.pressure_complex.shape == (2, 2, 2)
    assert result.cancelled is False
    assert result.requested_frequency_count == 2
    assert result.is_partial is False
    assert session.closed == 1


def test_completion_without_any_result_is_refused(run, config):
    """The injected sequence from the review: `completed`, nothing solved."""

    session = FakeSession([{"type": "completed"}])
    with pytest.raises(RuntimeError, match="completion with 0 of 1"):
        run(session, [1000.0], config)
    assert session.closed == 1


def test_completion_missing_one_frequency_is_refused(run, config):
    session = FakeSession([_result_event(500.0), {"type": "completed"}])
    with pytest.raises(RuntimeError, match="completion with 1 of 2"):
        run(session, [500.0, 1000.0], config)
    assert session.closed == 1


def test_cancellation_returns_an_explicitly_marked_partial_result(run, config):
    session = FakeSession([_result_event(500.0), {"type": "cancelled"}])
    result = run(session, [500.0, 1000.0], config)

    assert result.frequencies_hz.tolist() == [500.0]
    assert result.pressure_complex.shape == (1, 2, 2)
    assert result.cancelled is True
    assert result.requested_frequency_count == 2
    assert result.is_partial is True
    assert session.closed == 1


def test_cancellation_after_the_last_frequency_is_complete_but_marked(run, config):
    session = FakeSession([_result_event(500.0), {"type": "cancelled"}])
    result = run(session, [500.0], config)

    assert result.cancelled is True
    assert result.is_partial is False


def test_a_stream_that_ends_without_a_terminal_event_is_refused(run, config):
    session = FakeSession([_result_event(500.0)])
    with pytest.raises(RuntimeError, match="without a completed or cancelled"):
        run(session, [500.0, 1000.0], config)
    assert session.closed == 1


def test_an_empty_stream_is_refused_rather_than_returning_nothing(run, config):
    session = FakeSession([])
    with pytest.raises(RuntimeError, match="without a completed or cancelled"):
        run(session, [1000.0], config)
    assert session.closed == 1


def test_pressure_rows_that_do_not_match_the_angle_grid_are_refused(run, config):
    session = FakeSession([_result_event(500.0, angle_count=3), {"type": "completed"}])
    with pytest.raises(RuntimeError, match="angle grid does not match"):
        run(session, [500.0], config)
    assert session.closed == 1


def test_more_results_than_frequencies_are_refused(run, config):
    session = FakeSession(
        [_result_event(500.0), _result_event(1000.0), {"type": "completed"}]
    )
    with pytest.raises(RuntimeError, match="more results than requested"):
        run(session, [500.0], config)
    assert session.closed == 1


def test_a_frequency_echoed_out_of_order_is_refused(run, config):
    session = FakeSession([_result_event(1000.0), {"type": "completed"}])
    with pytest.raises(RuntimeError, match="out of order"):
        run(session, [500.0], config)
    assert session.closed == 1


def test_missing_initialization_metadata_closes_the_session(run, config):
    session = FakeSession([{"type": "completed"}], metadata={})
    session.metadata = None
    with pytest.raises(RuntimeError, match="no initialization metadata"):
        run(session, [1000.0], config)
    assert session.closed == 1


def test_metadata_without_angles_closes_the_session(run, config):
    session = FakeSession([{"type": "completed"}], metadata={"polar_angle_deg": []})
    with pytest.raises(RuntimeError, match="no observation angles"):
        run(session, [1000.0], config)
    assert session.closed == 1


def test_a_raising_progress_callback_closes_the_session(run, mesh_path):
    def explode(index, total, frequency_hz):
        raise KeyboardInterrupt("user quit mid-sweep")

    config = SolveConfig(
        observation=ObservationConfig(angle_count=2, angle_max_deg=90.0),
        progress_callback=explode,
    )
    session = FakeSession([_result_event(500.0), {"type": "completed"}])
    with pytest.raises(KeyboardInterrupt):
        run(session, [500.0], config)
    assert session.closed == 1


def test_a_raising_result_callback_closes_the_session(run, mesh_path):
    def explode(index, frequency_hz, entry):
        raise RuntimeError("consumer failed while plotting")

    config = SolveConfig(
        observation=ObservationConfig(angle_count=2, angle_max_deg=90.0),
        on_frequency_result=explode,
    )
    session = FakeSession([_result_event(500.0), {"type": "completed"}])
    with pytest.raises(RuntimeError, match="consumer failed"):
        run(session, [500.0], config)
    assert session.closed == 1


def test_a_result_callback_returning_false_asks_the_solver_to_stop(run, config):
    session = FakeSession(
        [_result_event(500.0), {"type": "cancelled"}]
    )
    config.on_frequency_result = lambda index, frequency_hz, entry: False
    result = run(session, [500.0, 1000.0], config)

    assert session.cancel_requests == 1
    assert result.cancelled is True
    assert result.is_partial is True


def test_the_returned_result_carries_the_normalized_directivity(run, config):
    session = FakeSession([_result_event(500.0), {"type": "completed"}])
    result = run(session, [500.0], config)

    # The fake's wire row is 1 and 0.5 Pa per unit velocity; the sweep
    # rescales it to unit acceleration, which moves the absolute level and
    # must leave the pattern alone.
    assert result.directivity_db[0, 0].tolist() == pytest.approx([0.0, -6.0205999132])
    absolute = 20.0 * np.log10(np.abs(result.pressure_complex) / 20e-6)
    assert result.spl_db == pytest.approx(absolute)
    assert result.spl_db[0, 0, 0] == pytest.approx(24.0364026333)


def test_an_ineffective_named_origin_is_refused_before_a_session_is_built(
    monkeypatch, mesh_path, config
):
    """The B2 guard at the sweep entry point, not only at construction.

    ``SolveConfig.__post_init__`` refuses a named observation origin without a
    frame, and ``tests/test_config.py`` covers that plus the helper being
    re-callable after mutation. What is only covered here is the ordering:
    ``solve_frequencies`` re-checks, and it must do so before it resolves
    Julia or constructs a session, or a caller who dropped the frame after
    construction pays for a worker start to be told no.
    """

    from hornlab_beat_bem import ObservationFrame

    # Construct the way a caller legitimately can, then drop the frame: this
    # is the mutation the constructor cannot see.
    config.observation.origin = "throat"
    config.frame_override = ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.zeros(3),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
    )
    config.frame_override = None

    def refuse_to_discover(*args, **kwargs):
        raise AssertionError("Julia was resolved before the origin was checked")

    def refuse_to_build(*args, **kwargs):
        raise AssertionError("a session was built before the origin was checked")

    monkeypatch.setattr(sweep, "discover_julia", refuse_to_discover)
    monkeypatch.setattr(sweep, "BeatSolveSession", refuse_to_build)
    with pytest.raises(NotImplementedError, match="needs an explicit frame_override"):
        solve_frequencies(mesh_path, [500.0], config)
