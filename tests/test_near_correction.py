"""The near-singular correction's core invariant, checked in the real solver."""

import subprocess
from pathlib import Path

import pytest

import hornlab_beat_bem as beat

SOLVER_DIR = Path(beat.__file__).parent / "julia"
SCRIPT = Path(__file__).parent / "near_correction_rule_independence.jl"

pytestmark = pytest.mark.slow


def test_full_correction_makes_assembly_independent_of_the_regular_rule():
    julia = beat.discover_julia()
    if julia is None:
        pytest.skip("no Julia executable (set HORNLAB_BEAT_JULIA)")
    completed = subprocess.run(
        [julia, f"--project={SOLVER_DIR}", str(SCRIPT), str(SOLVER_DIR)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS" in completed.stdout
