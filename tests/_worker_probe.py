"""A one-shot client of a persistent worker, used as a separate process.

Adoption is a property of *processes*, not of objects: the interesting
assertions are "a second interpreter finds the first one's runtime" and "a
client that dies mid-solve does not poison it". Both are unprovable from
inside a single test process, where the module-level worker cache would hand
back the same object either way. So the tests drive this file with
``subprocess`` and read the pids it prints.

Usage: ``_worker_probe.py <key.json> <adopt|solve|abandon>``. It prints one
JSON line on stdout and always detaches rather than shutting down, which is
the lifetime an application's quit hook is supposed to have.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from hornlab_beat_bem.worker_client import HostedBeatWorker


def _request_file(directory: Path, frequencies: list[float]) -> Path:
    path = directory / "request.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "config": {"tag_throat": 2},
                "cancel_path": str(directory / "cancel"),
                "frequencies_hz": frequencies,
            }
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    key = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    action = sys.argv[2] if len(sys.argv) > 2 else "adopt"
    worker = HostedBeatWorker(key)
    worker.ensure_started()
    report = {"host_pid": worker.host_pid, "engine_pid": worker.engine_pid}

    if action == "solve":
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            events = list(worker.submit(_request_file(directory, [100.0, 200.0])))
        report["events"] = [
            str(event.get("type"))
            for event in events
            if str(event.get("type")) != "status"
        ]
    elif action == "abandon":
        # Start a long job, get far enough in that the solve is genuinely
        # under way, then vanish without reading it or saying goodbye -- the
        # shape of an application that was killed mid-solve.
        directory = Path(tempfile.mkdtemp())
        stream = worker.submit(_request_file(directory, [100.0, 200.0, 300.0]))
        for event in stream:
            if str(event.get("type")) == "initialized":
                break
        print(json.dumps(report), flush=True)
        import os

        os._exit(0)

    print(json.dumps(report), flush=True)
    worker.detach()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
