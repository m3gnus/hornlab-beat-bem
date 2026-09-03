import os
import platform
import subprocess

from hornlab_beat_bem.worker import (
    _julia_process_env,
    _performance_core_count,
    _resolve_julia_threads,
)


def test_explicit_thread_counts_pass_through():
    assert _resolve_julia_threads(4) == "4"
    assert _resolve_julia_threads("6") == "6"
    assert _resolve_julia_threads(0) == "1"


def test_unparseable_thread_count_falls_back_to_the_auto_answer():
    assert _resolve_julia_threads("many") == str(_performance_core_count())


def test_auto_targets_performance_cores_not_every_core():
    resolved = int(_resolve_julia_threads("auto"))
    assert 1 <= resolved <= (os.cpu_count() or 1)
    assert resolved == _performance_core_count()


def test_performance_core_count_matches_sysctl_on_asymmetric_apple_silicon():
    """The whole point of the change, asserted only where it applies."""
    if platform.system() != "Darwin":
        return
    probe = subprocess.run(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return  # symmetric part; the full count is already correct
    performance = int(probe.stdout.strip())
    assert _performance_core_count() == performance
    if performance < (os.cpu_count() or 1):
        assert int(_resolve_julia_threads("auto")) < (os.cpu_count() or 1)


def test_metal_sweep_pipelining_is_left_for_the_solver_to_decide(monkeypatch):
    """The worker starts before any mesh is known, so it must not answer this.

    Setting a value here would pin one answer for every solve the worker goes
    on to run, and the right answer depends on the mesh: see
    `metal_pipeline_requested` in julia/BeatEngineDriver.jl.
    """
    monkeypatch.delenv("BLAB_METAL_PIPELINE", raising=False)
    env = _julia_process_env("auto", None)
    assert "BLAB_METAL_PIPELINE" not in env


def test_an_explicit_metal_sweep_pipelining_choice_still_reaches_the_solver(monkeypatch):
    for value in ("0", "1"):
        monkeypatch.setenv("BLAB_METAL_PIPELINE", value)
        env = _julia_process_env("auto", None)
        assert env["BLAB_METAL_PIPELINE"] == value
