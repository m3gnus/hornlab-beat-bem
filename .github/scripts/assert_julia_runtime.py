"""Fail the build when a runner that should have Julia silently does not.

Three test files in ``tests/`` -- ``test_solve_smoke.py``,
``test_near_correction.py`` and ``test_engine_bundles.py`` -- take a ``julia``
fixture that calls ``pytest.skip`` when ``discover_julia()`` returns ``None``.
That guard is right for a developer without the runtime installed. In CI it is
not: it turns a missing Julia into three skips and a green badge over a suite
that never ran a single BEM solve. Measured on this repository, ``pytest -m
slow`` with no Julia on PATH reports ``3 skipped`` and exits 0.

So resolve Julia the way the package itself does, prove the executable runs,
and print what was found. CI calls this before pytest so a lost runtime reads
as one legible error rather than as skips nobody reads.
"""

from __future__ import annotations

import subprocess
import sys

from hornlab_beat_bem import runtime


def main() -> int:
    executable = runtime.discover_julia()
    if executable is None:
        print(
            "No Julia executable was found on a runner that is supposed to "
            "provide one.\n"
            "\n"
            "The slow tests would skip themselves and this job would pass "
            "while running no\n"
            "BEM solve at all, so it fails here instead. Discovery order is "
            "explicit argument,\n"
            f"then ${runtime.JULIA_ENV_VAR}, then the provisioned runtime "
            "directory, then PATH.",
            file=sys.stderr,
        )
        return 1

    try:
        version = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        print(
            f"Julia was discovered at {executable} but does not run: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"Julia is available; the slow tests will run for real.\n  path    {executable}\n  version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
