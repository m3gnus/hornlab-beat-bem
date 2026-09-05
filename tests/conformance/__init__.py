"""Named conformance cases for the BEAT package, and the runner that records them.

This lives under ``tests/`` rather than inside ``hornlab_beat_bem`` on purpose.
The capability report (``hornlab_beat_bem.capabilities``) is a product contract
a consumer reads; this harness is verification machinery that runs the contract
against a real solver and writes an evidence record. Shipping it in the wheel
would put the second thing in a consumer's import path without giving them
anything to call.

The consequence is that the harness must never assume it is the only copy of
the package on the machine. Every record it writes carries the loaded package
path, its git SHA, the interpreter and the environment prefix precisely so a
run against a stale checkout or a drifted virtual environment is visible in the
record rather than inferred afterwards from a number that looks wrong.
"""
