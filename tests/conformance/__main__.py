"""Run the conformance cases outside pytest.

    PYTHONPATH=tests python -m conformance --list
    PYTHONPATH=tests python -m conformance --out <results directory>

The records go where ``--out`` says and nowhere by default that is outside the
working directory: they carry interpreter paths, environment overrides and a
registry directory, which belong in a local evidence directory rather than in
anything committed.
"""

from __future__ import annotations

import sys

from .harness import main

if __name__ == "__main__":
    sys.exit(main())
