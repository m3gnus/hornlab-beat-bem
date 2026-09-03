"""Turn a pytest JUnit report into a count, and refuse a vacuous green.

A CI job that reports "passed" tells you an exit code, not a coverage. The
failure this guards against is the one metal-bem's badge had for months: green
over a suite that quietly stopped running the part that mattered. Here that
shape is reachable in one step -- lose the Julia runtime and every test that
performs a real solve skips itself, leaving 60 passed, 3 skipped, exit 0.

So assert the shape of the run rather than only its exit code:

* at least ``--min-tests`` tests were collected, which catches a collection
  error or a path that silently matches nothing;
* nothing skipped, because in this repository every skip is a lost solve --
  there is no ``skipif`` on platform anywhere in ``tests/``, so a legitimate
  skip does not currently exist and a new one should have to be argued for
  here rather than absorbed silently;
* nothing failed or errored, which pytest's exit code already covers and this
  restates so that the printed summary is the whole verdict.

It prints the per-outcome counts either way, so the job log carries the number
a report can quote instead of the word "green".
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="pytest --junitxml output")
    parser.add_argument(
        "--min-tests",
        type=int,
        default=1,
        help="fail if fewer than this many tests were collected",
    )
    arguments = parser.parse_args()

    if not arguments.report.is_file():
        print(f"No pytest report at {arguments.report}.", file=sys.stderr)
        return 1

    root = ElementTree.parse(arguments.report).getroot()
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

    total = failures = errors = skipped = 0
    skip_reasons: list[str] = []
    for suite in suites:
        total += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
        for case in suite.iter("testcase"):
            for skip in case.findall("skipped"):
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                skip_reasons.append(f"{name}: {skip.get('message', '')}")

    passed = total - failures - errors - skipped
    print(
        f"pytest: {passed} passed, {failures} failed, {errors} errored, "
        f"{skipped} skipped, {total} collected"
    )

    problems: list[str] = []
    if total < arguments.min_tests:
        problems.append(
            f"only {total} tests were collected, expected at least "
            f"{arguments.min_tests} -- collection is broken or the path "
            f"matched nothing"
        )
    if failures or errors:
        problems.append(f"{failures} failed and {errors} errored")
    if skipped:
        problems.append(
            f"{skipped} tests skipped, and every skip in this repository is a "
            f"lost solve:\n    " + "\n    ".join(skip_reasons)
        )

    if problems:
        print("\nThis run does not prove what a green job would claim:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
