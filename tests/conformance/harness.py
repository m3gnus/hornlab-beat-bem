"""The conformance runner: run a named case, record what actually ran.

A conformance record answers "what did this number come from", and it has to
answer it without the session that produced it. That is why the record carries
the loaded package path and git SHA, the interpreter and environment prefix,
the Julia version, the mesh checksum, the explicit frequency list and the
worker's cold/warm state rather than a summary sentence: every one of those has
been, in this workspace, the actual explanation for a result that looked like a
regression. A drifted virtual environment holding a package copy without the
Metal backend is not visible in a directivity plot.

Wall timings are recorded for the same reason -- provenance, not benchmark. The
record also captures the host's load average at sample time, because a timing
taken next to another agent's test suite is not a measurement of this code.

The comparison against ``hornlab-metal-bem`` is recorded the same way. It runs
whenever that package is importable and its solve can start here, and otherwise
records an explicit ``skip_reason``. It is never a pytest skip: the CI report
checker rejects skips outright, and "hornlab-metal-bem is not installed" is the
normal state of a CI runner rather than a suite that did not run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import hornlab_beat_bem as beat
from hornlab_beat_bem import runtime as beat_runtime
from hornlab_beat_bem import worker as beat_worker
from hornlab_beat_bem import worker_client, worker_registry
from hornlab_beat_bem.capabilities import CAPABILITY_SCHEMA_VERSION
from hornlab_beat_bem.config import SolveConfig
from hornlab_beat_bem.result import SolveResult

#: Shape version of the record :func:`run_case` writes. Bump on a structural
#: change, so a stored record can be read by the reader that understands it.
CONFORMANCE_RECORD_SCHEMA_VERSION = 1

#: Environment variables that change what a solve does or which runtime runs
#: it. Recorded by name and value, because "it was set on that machine" is the
#: explanation more often than anything in the source.
_RECORDED_ENV_PREFIXES = ("HORNLAB_BEAT_", "BLAB_", "JULIA_")


MeshBuilder = Callable[[Path], Path]
ConfigBuilder = Callable[[str], SolveConfig]
Acceptance = Callable[[SolveResult], dict[str, Any]]
StaticCheck = Callable[[], dict[str, Any]]
MetalConfigBuilder = Callable[[], Any]
#: ``(beat_result, counterpart_result, comparison_entry) -> None``. The scorer
#: is handed the record's comparison entry and writes its metrics into it,
#: rather than returning them, so a band assertion that fires cannot take the
#: numbers with it. See :func:`run_case`.
Comparison = Callable[[SolveResult, Any, dict[str, Any]], None]


@dataclass(frozen=True)
class ConformanceCase:
    """One named, explicitly parameterised conformance case.

    A case is either *static* -- it asserts something about the package's
    declared contract and runs no solver -- or a *solve* case, which builds a
    mesh and a config, runs the BEAT package, and scores the result. Both kinds
    produce the same record shape, so a run's evidence is one directory of
    comparable files.
    """

    name: str
    description: str
    backend: str = beat.BEAT_CPU
    frequencies_hz: tuple[float, ...] = ()
    make_mesh: MeshBuilder | None = None
    make_config: ConfigBuilder | None = None
    accept: Acceptance | None = None
    static_check: StaticCheck | None = None
    make_metal_bem_config: MetalConfigBuilder | None = None
    compare: Comparison | None = None
    tags: tuple[str, ...] = ()
    #: Extra case parameters worth recording verbatim (sphere radius, distance,
    #: subdivision level). Kept beside the metrics so a record is reproducible
    #: without reading this file.
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_solver(self) -> bool:
        return self.make_config is not None


def _git_description(package_dir: Path) -> dict[str, Any]:
    """The commit the loaded package came from, when it came from a checkout."""

    def _git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(package_dir), *args],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None, "branch": None, "source": "not a git checkout"}
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "source": "git checkout",
    }


def julia_version(executable: str | None) -> str | None:
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def worker_state() -> dict[str, Any]:
    """Cold or warm, and by what evidence.

    ``cold`` means no persistent host is running in the registry this process
    would use and this process holds no worker of its own, so the next solve
    pays a Julia start and its compilation. Anything else is ``warm`` and its
    timings are not comparable with a cold one.
    """

    try:
        directory = worker_registry.worker_dir()
        live = len(worker_client.find_live_hosts(directory))
        directory_name: str | None = str(directory)
    except Exception as exc:  # pragma: no cover - registry refusal path
        live = 0
        directory_name = f"<unavailable: {type(exc).__name__}>"
    in_process = len(getattr(beat_worker, "_WORKERS", {}))
    return {
        "state": "cold" if live == 0 and in_process == 0 else "warm",
        "live_registry_hosts": live,
        "in_process_workers": in_process,
        "registry_dir": directory_name,
        "persistent_host_enabled": worker_registry.persistent_host_enabled(),
    }


def _load_average() -> list[float] | None:
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        return None


def provenance(*, julia_executable: str | None = None) -> dict[str, Any]:
    """Everything needed to say which code, on which host, produced a record."""

    package_dir = Path(beat_runtime.PACKAGE_DIR)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "record_schema_version": CONFORMANCE_RECORD_SCHEMA_VERSION,
        "package": {
            "name": "hornlab-beat-bem",
            "version": beat.package_version(),
            "path": str(package_dir),
            "fingerprint": beat_runtime.package_fingerprint(),
            "git": _git_description(package_dir.parent),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "in_virtual_environment": sys.prefix != sys.base_prefix,
        },
        "julia": {
            "executable": julia_executable,
            "version": julia_version(julia_executable),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "environment_overrides": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(_RECORDED_ENV_PREFIXES)
        },
        "load_average_at_sample_time": _load_average(),
    }


def mesh_fingerprint(mesh_path: Path, *, scale: float = 1.0) -> dict[str, Any]:
    """Checksum plus the facts the solve actually depends on."""

    digest = hashlib.sha256()
    with Path(mesh_path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    info = beat.read_gmsh22_info(mesh_path, scale=scale)
    return {
        "file_name": Path(mesh_path).name,
        "sha256": digest.hexdigest(),
        "bytes": Path(mesh_path).stat().st_size,
        "scale": float(scale),
        "n_vertices": info.n_vertices,
        "n_triangles": info.n_triangles,
        "physical_tag_areas_m2": {
            str(tag): float(area)
            for tag, area in sorted(info.physical_tag_areas_m2.items())
        },
    }


def _solver_facts(result: SolveResult, requested: int) -> dict[str, Any]:
    solved = int(np.asarray(result.frequencies_hz).size)
    if result.cancelled:
        status = "cancelled"
    elif solved == requested:
        status = "completed"
    else:  # pragma: no cover - sweep raises before this is reachable
        status = "truncated"
    diagnostics = [entry.get("native_diagnostics") for entry in result.solver_log]
    return {
        "completion_status": status,
        "cancelled": bool(result.cancelled),
        "is_partial": bool(result.is_partial),
        "requested_frequency_count": result.requested_frequency_count,
        "solved_frequency_count": solved,
        "solved_frequencies_hz": [float(value) for value in result.frequencies_hz],
        "observation_planes": list(result.observation_planes),
        "observation_angle_count": int(
            np.asarray(result.observation_angles_deg).size
        ),
        "timings_s": {key: float(value) for key, value in result.timings.items()},
        "native_diagnostics": [entry for entry in diagnostics if entry is not None],
    }


def _metal_bem_run(
    case: ConformanceCase,
    mesh_path: Path | None,
    frequencies: tuple[float, ...],
    beat_result: SolveResult | None,
) -> tuple[dict[str, Any], Any]:
    """Run the same case through hornlab-metal-bem, or say why it did not.

    Returns the record entry and the counterpart's result, and deliberately
    does **not** score them: the scoring is where a case can fail, so the
    caller writes this entry into the record first and hands the same entry to
    ``case.compare``, which fills in its ``metrics`` before asserting
    anything. That ordering is what keeps a disagreement from destroying the
    evidence that shows it -- both the entry and the numbers in it.

    Read-only use of that package: this imports it and calls its public solve.
    It never writes to its repository.
    """

    header: dict[str, Any] = {"package": "hornlab-metal-bem"}
    if case.make_metal_bem_config is None or case.compare is None:
        return {
            **header,
            "ran": False,
            "skip_reason": "this case defines no hornlab-metal-bem counterpart",
        }, None
    if beat_result is None or mesh_path is None:
        return {
            **header,
            "ran": False,
            "skip_reason": "the BEAT side of this case produced no result to compare",
        }, None
    try:
        import hornlab_metal_bem as metal
    except Exception as exc:
        return {
            **header,
            "ran": False,
            "skip_reason": (
                "hornlab-metal-bem is not importable in this environment "
                f"({type(exc).__name__}: {exc})"
            ),
        }, None
    started = time.monotonic()
    try:
        config = case.make_metal_bem_config()
        metal_result = metal.solve_frequencies(str(mesh_path), list(frequencies), config)
    except Exception as exc:
        return {
            **header,
            "ran": False,
            "path": str(Path(metal.__file__).resolve().parent),
            "skip_reason": (
                "hornlab-metal-bem is importable but its solve did not run here "
                f"({type(exc).__name__}: {str(exc).splitlines()[0][:200]})"
            ),
            "wall_seconds": round(time.monotonic() - started, 3),
        }, None
    return {
        **header,
        "ran": True,
        "skip_reason": None,
        "path": str(Path(metal.__file__).resolve().parent),
        "version": getattr(metal, "__version__", None),
        "wall_seconds": round(time.monotonic() - started, 3),
    }, metal_result


def run_case(
    case: ConformanceCase,
    *,
    work_dir: Path,
    output_dir: Path | None = None,
    julia_executable: str | None = None,
) -> dict[str, Any]:
    """Run one case and return its record, writing it when ``output_dir`` is set.

    Acceptance failures are raised, not swallowed: a conformance case that
    fails must fail its caller. The record is written first, so the evidence
    survives the failure.

    The cross-package comparison runs inside the guarded section too, because
    it is part of the case: a disagreement outside the recorded band is a
    failure, and running it in the ``finally`` block would have let it destroy
    the record it was supposed to appear in. When the case has already failed,
    the comparison is still recorded -- as data, never as a second exception
    that would mask the first.

    ``case.compare`` is handed the comparison entry itself and populates it,
    rather than returning a table this function assigns afterwards: the
    assignment is the step a band assertion used to skip, leaving a
    ``ran: true`` entry with no ``metrics``.
    """

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    julia = julia_executable or beat.discover_julia()

    record: dict[str, Any] = {
        "schema_version": CONFORMANCE_RECORD_SCHEMA_VERSION,
        "case": {
            "name": case.name,
            "description": case.description,
            "backend": case.backend,
            "kind": "solve" if case.needs_solver else "static",
            "frequencies_hz": [float(value) for value in case.frequencies_hz],
            "frequency_count": len(case.frequencies_hz),
            "tags": list(case.tags),
            "parameters": dict(case.parameters),
        },
        "provenance": provenance(julia_executable=julia),
        "worker_state_before": worker_state(),
    }

    mesh_path: Path | None = None
    result: SolveResult | None = None
    failure: BaseException | None = None
    started = time.monotonic()
    try:
        if case.static_check is not None:
            record["metrics"] = case.static_check()
        if case.needs_solver:
            if julia is None:
                raise beat.BeatUnavailableError(
                    "no Julia executable was found; set HORNLAB_BEAT_JULIA"
                )
            if case.make_mesh is None:
                raise ValueError(f"case {case.name!r} solves but builds no mesh")
            mesh_path = Path(case.make_mesh(work_dir))
            config = case.make_config(julia)
            record["mesh"] = mesh_fingerprint(mesh_path, scale=config.mesh_scale)
            record["request"] = {
                "backend": config.beat_backend,
                "solve_precision": config.solve_precision,
                "source_tag": config.source_tag,
                "source_motion": config.source_motion,
                "source_count": len(config.velocity_sources),
                "native_symmetry_plane": config.native_symmetry_plane,
                "solver_image_mode": beat.beat_image_mode(config),
                "ground_plane_enabled": beat.ground_plane_enabled(config),
                "near_correction": config.near_correction,
                "surface_traces": config.surface_traces,
                "quadrature_order": config.quadrature_order,
                "singular_order": config.singular_order,
                "observation_planes": list(config.observation.planes),
                "observation_angle_count": config.observation.angle_count,
                "observation_distance_m": config.observation.distance_m,
                "observation_point_count": (
                    len(config.observation.planes) * config.observation.angle_count
                ),
                "air_density_kg_m3": config.air_density,
                "sound_speed_m_s": config.sound_speed,
            }
            solve_started = time.monotonic()
            result = beat.solve_frequencies(mesh_path, case.frequencies_hz, config)
            record["solve_wall_seconds"] = round(time.monotonic() - solve_started, 3)
            record["solver"] = _solver_facts(result, len(case.frequencies_hz))
            if case.accept is not None:
                record["metrics"] = case.accept(result)
        comparison, metal_result = _metal_bem_run(
            case, mesh_path, tuple(case.frequencies_hz), result
        )
        # Written into the record *before* it is scored, and the same dict is
        # what the scorer fills in, so the numbers a disagreement is argued
        # from survive the assertion that rejects them.
        record["comparison"] = comparison
        if metal_result is not None and case.compare is not None:
            case.compare(result, metal_result, comparison)
        record["status"] = "passed"
    # BaseException on purpose, and re-raised below: a KeyboardInterrupt or a
    # pytest outcome must still leave the evidence of what had run, and must
    # still reach the caller unchanged.
    except BaseException as exc:  # recorded, then re-raised
        failure = exc
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:400]}"
    finally:
        record["wall_seconds"] = round(time.monotonic() - started, 3)
        record["worker_state_after"] = worker_state()
        if "comparison" not in record:
            # The case has already failed. Say why the comparison could not be
            # made, without replacing that failure with a second one.
            try:
                record["comparison"] = _metal_bem_run(
                    case, mesh_path, tuple(case.frequencies_hz), result
                )[0]
            except BaseException as exc:  # noqa: BLE001 - recorded, not raised
                record["comparison"] = {
                    "package": "hornlab-metal-bem",
                    "ran": False,
                    "skip_reason": (
                        "the comparison raised while recording an already "
                        f"failed case ({type(exc).__name__}: "
                        f"{str(exc).splitlines()[0][:200]})"
                    ),
                }
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{case.name}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
    if failure is not None:
        raise failure
    return record


def main(argv: list[str] | None = None) -> int:
    """Run every case (or a named subset) and write the records to a directory."""

    import argparse
    import tempfile

    from .cases import all_cases

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="conformance-results",
        help="directory for the JSON records (default: ./conformance-results)",
    )
    parser.add_argument(
        "--case", action="append", default=None, help="run only this case (repeatable)"
    )
    parser.add_argument("--list", action="store_true", help="list the case names and exit")
    arguments = parser.parse_args(argv)

    cases = all_cases()
    if arguments.list:
        for case in cases:
            print(f"{case.name}\t{case.description}")
        return 0
    if arguments.case:
        wanted = set(arguments.case)
        unknown = wanted - {case.name for case in cases}
        if unknown:
            parser.error("unknown case(s): " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case.name in wanted]

    failures = 0
    with tempfile.TemporaryDirectory(prefix="beat-conformance-") as work_dir:
        for case in cases:
            try:
                run_case(case, work_dir=Path(work_dir), output_dir=Path(arguments.out))
            except BaseException as exc:
                failures += 1
                print(f"FAIL {case.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            else:
                print(f"PASS {case.name}")
    return 1 if failures else 0
