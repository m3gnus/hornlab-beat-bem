"""A stand-in for ``solver.jl --worker`` that speaks the same JSON lines.

The persistent-worker machinery is about process lifetime, adoption and
crash recovery; none of that is about Julia, and pinning those tests to a
real Julia start would put a 3-15 s cold start in front of every one of them
and make them skip wherever Julia is absent. This script is the same protocol
in ~60 lines, so the lifecycle suite runs in seconds and in CI on any host.

It is driven exactly as the real worker is -- executable, script path,
``--worker`` -- so the command construction, the pipes, the event parsing and
the cancellation file are the production ones. What it does *not* exercise is
the solver, which ``test_solve_smoke.py`` and the slow marker already cover.

Knobs, read from the environment because that is what the host inherits:
``FAKE_BEAT_BOOT_S`` delays the ready event (a slow cold start),
``FAKE_BEAT_SOLVE_S`` delays each frequency (a long job to interrupt),
``FAKE_BEAT_FAIL`` makes startup fail.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def emit(**payload: object) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def solve(request: dict) -> None:
    frequencies = [float(value) for value in request.get("frequencies_hz", [1000.0])]
    config = request.get("config", {})
    cancel_path = request.get("cancel_path")
    per_frequency = float(os.environ.get("FAKE_BEAT_SOLVE_S", "0"))
    angles = [0.0, 45.0, 90.0]
    emit(type="initialized", polar_angle_deg=angles, worker_pid=os.getpid())
    for frequency in frequencies:
        if per_frequency > 0.0:
            time.sleep(per_frequency)
        if cancel_path and Path(cancel_path).exists():
            emit(type="cancelled")
            return
        row = {"real": [[1.0] * len(angles)], "imag": [[0.0] * len(angles)]}
        emit(
            type="result",
            result={
                "freq_hz": frequency,
                "channel_names": ["main"],
                "horizontal_pressure": row,
                "vertical_pressure": row,
                "impedance": [[1.0, -1.0]],
                "timings": {"assembly_s": 0.0, "solve_s": 0.0, "field_s": 0.0},
                "source_tag": config.get("tag_throat"),
            },
        )
    emit(type="completed", solved_count=len(frequencies))


def main() -> int:
    if os.environ.get("FAKE_BEAT_FAIL", "") == "1":
        emit(type="failed", error="fake worker was told to fail")
        return 1
    boot = float(os.environ.get("FAKE_BEAT_BOOT_S", "0"))
    if boot > 0.0:
        time.sleep(boot)
    if "--worker" not in sys.argv[1:]:
        request_path = None
        arguments = sys.argv[1:]
        for index, argument in enumerate(arguments):
            if argument == "--request" and index + 1 < len(arguments):
                request_path = arguments[index + 1]
        if request_path is None:
            emit(type="failed", error="fake worker needs --worker or --request")
            return 1
        solve(json.loads(Path(request_path).read_text(encoding="utf-8")))
        return 0
    emit(type="ready", protocol="boundary_lab_julia_worker", pid=os.getpid())
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            message = json.loads(text)
            request = json.loads(
                Path(str(message["request"])).read_text(encoding="utf-8")
            )
            solve(request)
        except Exception as exc:  # noqa: BLE001 - mirrors the driver's own catch
            emit(type="failed", error=f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
