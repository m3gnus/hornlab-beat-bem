"""Run BEAT Engine sweeps and map them onto HornLab's native result shape.

This module is the convention boundary. The vendored Julia solver:

- drives the source tag with ``q = i*rho*omega*v_n`` on a **1 m/s normal
  velocity** basis (e^{-i omega t} time convention, e^{+ikr} outgoing waves);
- observes polar cuts around the **global mesh origin** with the forward axis
  fixed to +z (horizontal cut in the x-z plane, vertical in y-z);
- reports a legacy-scaled radiation-impedance pair per radiator:
  ``[Re(F)/2, -Im(F)/2]`` where ``F = 10 * sym_factor * sum(p_avg * area) / v``.

Everything returned from here uses the shared HornLab convention instead:
pressures and the throat mean pressure are per **unit normal acceleration**
(``p_accel = p_vel / (-i*omega)``), and the observation origin is whatever
frame the caller supplied (applied as a rigid mesh translation).
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from .config import (
    ObservationFrame,
    SolveConfig,
    beat_symmetry_mode,
    reject_unsupported_native_symmetry,
)
from .mesh import read_gmsh22_info
from .result import SolveResult
from .runtime import DEFAULT_SOLVER_SCRIPT, discover_julia
from .worker import BeatSolveSession, get_worker

#: The solver multiplies the integrated source force by this legacy display
#: factor before dividing by the drive; undone when reconstructing pressure.
_BEAT_IMPEDANCE_FORCE_FACTOR = 10.0

_SYMMETRY_FACTOR = {"off": 1, "x": 2, "xy": 4}

_FRAME_AXIS_TOLERANCE = 1.0e-3


class BeatUnavailableError(RuntimeError):
    """The BEAT runtime (Julia, solver assets) cannot run a solve."""


def _validated_frame_translation(frame: ObservationFrame | None) -> tuple[float, float, float]:
    """Return the mesh translation that puts the frame origin at the solver origin.

    The BEAT solver's observation geometry is hard-wired to the +z axis with
    horizontal along +x, so a frame is only representable when its basis is
    axis-aligned; anything else must be refused rather than silently measured
    on the wrong arc.
    """

    if frame is None:
        return (0.0, 0.0, 0.0)
    axis = np.asarray(frame.axis, dtype=float).reshape(3)
    u = np.asarray(frame.u, dtype=float).reshape(3)
    v = np.asarray(frame.v, dtype=float).reshape(3)
    expected = {
        "axis": np.array([0.0, 0.0, 1.0]),
        "u": np.array([1.0, 0.0, 0.0]),
        "v": np.array([0.0, 1.0, 0.0]),
    }
    for name, (vector, target) in zip(expected, ((axis, expected["axis"]), (u, expected["u"]), (v, expected["v"]))):
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"observation frame {name} must be a finite non-zero vector")
        if float(np.linalg.norm(vector / norm - target)) > _FRAME_AXIS_TOLERANCE:
            raise NotImplementedError(
                "hornlab-beat-bem requires an axis-aligned observation frame "
                f"(+z forward, +x horizontal); frame {name}={vector.tolist()} is tilted"
            )
    origin = np.asarray(frame.origin, dtype=float).reshape(3)
    if not np.all(np.isfinite(origin)):
        raise ValueError("observation frame origin must be finite")
    return (float(-origin[0]), float(-origin[1]), float(-origin[2]))


def _request_payload(
    mesh_path: str | Path,
    frequencies: np.ndarray,
    config: SolveConfig,
    translation: tuple[float, float, float],
) -> dict[str, Any]:
    observation = config.observation
    solver_config: dict[str, Any] = {
        "mesh_file": str(mesh_path),
        "scale_factor": float(config.mesh_scale),
        "meshes": [
            {
                "name": "mesh",
                "file": str(mesh_path),
                "scale_factor": float(config.mesh_scale),
                "translation_m": list(translation),
            }
        ],
        "distance": float(observation.distance_m),
        "axial_offset": 0.0,
        "step_size": float(observation.step_deg),
        "min_angle": float(observation.angle_min_deg),
        "max_angle": float(observation.angle_max_deg),
        "freq_min": float(frequencies.min()),
        "freq_max": float(frequencies.max()),
        "freq_count": int(frequencies.size),
        "tag_throat": config.source_tag,
        "rho": float(config.air_density),
        "sound_speed": float(config.sound_speed),
        "symmetry": beat_symmetry_mode(config.native_symmetry_plane),
        "flat_target_normalization_enabled": False,
        "spherical_sampling_enabled": False,
        "quadrature_order": int(config.quadrature_order),
        "singular_order": int(config.singular_order),
    }
    if config.regular_quadrature_mode is not None:
        solver_config["regular_quadrature_mode"] = config.regular_quadrature_mode
    return {
        "schema_version": 2,
        "config": solver_config,
        "frequencies_hz": [float(value) for value in frequencies],
    }


def _wire_rows(raw: dict[str, Any] | None) -> np.ndarray:
    if raw is None:
        raise ValueError("BEAT solver result is missing a pressure block")
    real = np.asarray(raw["real"], dtype=np.float64)
    imag = np.asarray(raw["imag"], dtype=np.float64)
    return real + 1j * imag


def generated_frequency_grid(config: SolveConfig) -> np.ndarray:
    if config.freq_count == 1:
        return np.asarray([config.freq_min_hz], dtype=np.float64)
    if config.freq_spacing == "linear":
        return np.linspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)
    return np.geomspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)


def solve(mesh_path: str | Path, config: SolveConfig) -> SolveResult:
    """Solve the config's generated frequency grid."""

    return solve_frequencies(mesh_path, generated_frequency_grid(config), config)


def solve_frequencies(
    mesh_path: str | Path,
    frequencies_hz: Iterable[float],
    config: SolveConfig,
    *,
    status_callback: Any = None,
) -> SolveResult:
    """Solve an explicit frequency list on the configured backend.

    The list is solved in the given order (callers may pass a live-plotting
    execution order); the returned arrays follow that order too, so callers
    wanting the canonical ascending axis sort the result afterwards.
    """

    reject_unsupported_native_symmetry(config)
    frequencies = np.asarray(list(frequencies_hz), dtype=np.float64)
    if frequencies.size == 0:
        raise ValueError("frequencies_hz must contain at least one frequency")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must be finite and positive")
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError("frequencies_hz must not contain duplicates")

    julia = discover_julia(config.julia_executable)
    if julia is None:
        raise BeatUnavailableError(
            "No Julia executable was found. Install Julia or set HORNLAB_BEAT_JULIA."
        )
    if not DEFAULT_SOLVER_SCRIPT.exists():
        raise BeatUnavailableError(
            f"Bundled BEAT solver script is missing: {DEFAULT_SOLVER_SCRIPT}"
        )

    mesh_info = read_gmsh22_info(mesh_path, scale=config.mesh_scale)
    source_area = mesh_info.physical_tag_areas_m2.get(config.source_tag, 0.0)
    if source_area <= 0.0:
        raise ValueError(
            f"mesh {mesh_path} has no triangles with source tag {config.source_tag}"
        )
    symmetry_mode = beat_symmetry_mode(config.native_symmetry_plane)
    symmetry_factor = _SYMMETRY_FACTOR[symmetry_mode]

    translation = _validated_frame_translation(config.frame_override)
    payload = _request_payload(mesh_path, frequencies, config, translation)

    started = time.time()
    session = BeatSolveSession(
        payload,
        julia_executable=julia,
        beat_backend=config.beat_backend,
        julia_project=None if config.julia_project is None else Path(config.julia_project),
        julia_threads=config.julia_threads,
        julia_sysimage=None if config.julia_sysimage is None else Path(config.julia_sysimage),
        persistent_worker=config.persistent_worker,
        status_callback=status_callback,
    )
    assert session.metadata is not None
    angles = np.asarray(session.metadata.get("polar_angle_deg", []), dtype=np.float64)
    if angles.size == 0:
        raise RuntimeError("BEAT solver reported no observation angles")

    planes = list(config.observation.planes)
    total = int(frequencies.size)
    pressure = np.full((total, len(planes), angles.size), np.nan, dtype=np.complex128)
    spl = np.full((total, len(planes), angles.size), np.nan, dtype=np.float64)
    impedance = np.full(total, np.nan, dtype=np.complex128)
    solved_frequencies = np.full(total, np.nan, dtype=np.float64)
    timings = {"assembly_s": 0.0, "solve_s": 0.0, "field_s": 0.0}
    solver_log: list[dict[str, Any]] = []
    solved_count = 0
    cancelled = False

    for event in session.events():
        event_type = str(event.get("type", ""))
        if event_type == "cancelled":
            cancelled = True
            break
        if event_type == "completed":
            break
        if event_type != "result":
            continue

        raw = event["result"]
        index = solved_count
        if index >= total:
            raise RuntimeError("BEAT solver returned more results than requested frequencies")
        frequency = float(raw["freq_hz"])
        omega = 2.0 * math.pi * frequency
        # 1 m/s velocity basis -> unit normal acceleration (v = a/(-i*omega)).
        acceleration_scale = 1.0 / (-1j * omega)

        channel_names = list(raw.get("channel_names") or ["main"])
        if len(channel_names) != 1:
            raise RuntimeError(
                "hornlab-beat-bem drives a single synthesized channel; the "
                f"solver returned channels {channel_names}"
            )
        rows_by_cut = {
            "horizontal": _wire_rows(raw.get("horizontal_pressure"))[0],
            "vertical": _wire_rows(raw.get("vertical_pressure"))[0],
        }
        entry_pressure = np.stack(
            [rows_by_cut[plane] * acceleration_scale for plane in planes]
        )
        if entry_pressure.shape[1] != angles.size:
            raise RuntimeError(
                "BEAT solver angle grid does not match its pressure rows: "
                f"{entry_pressure.shape[1]} != {angles.size}"
            )
        with np.errstate(divide="ignore"):
            entry_spl = 20.0 * np.log10(np.abs(entry_pressure) / 20e-6)

        impedance_pairs = raw.get("impedance") or []
        entry_impedance: complex | None = None
        if impedance_pairs:
            pair = impedance_pairs[0]
            if len(pair) == 2 and all(math.isfinite(float(value)) for value in pair):
                # Undo the solver's [Re(F)/2, -Im(F)/2] packing and its x10
                # display factor. The wire force integrates the full domain
                # (reduced-mesh sum times the symmetry factor), so dividing by
                # the same factor and the reduced-mesh source area yields the
                # area-weighted mean pressure per unit velocity.
                force = 2.0 * float(pair[0]) - 2.0j * float(pair[1])
                mean_pressure_velocity = force / (
                    _BEAT_IMPEDANCE_FORCE_FACTOR * symmetry_factor * source_area
                )
                entry_impedance = complex(mean_pressure_velocity * acceleration_scale)

        pressure[index] = entry_pressure
        spl[index] = entry_spl
        impedance[index] = entry_impedance if entry_impedance is not None else complex(np.nan, np.nan)
        solved_frequencies[index] = frequency
        raw_timings = raw.get("timings") or {}
        for key in timings:
            timings[key] += float(raw_timings.get(key, 0.0))

        log_entry: dict[str, Any] = {
            "frequency_hz": frequency,
            "converged": True,
            "observation_angles_deg": angles,
            "observation_planes": planes,
            "observation_pressure_complex": entry_pressure,
            "observation_spl_db": entry_spl,
            "impedance": entry_impedance,
            "timings": dict(raw_timings),
            "native_diagnostics": raw.get("diagnostics"),
        }
        solver_log.append(log_entry)
        solved_count += 1

        if config.progress_callback is not None:
            config.progress_callback(index, total, frequency)
        if config.on_frequency_result is not None:
            keep_going = config.on_frequency_result(index, frequency, log_entry)
            if keep_going is False:
                session.request_cancel()

    del cancelled  # partial results are represented by truncation below
    timings["total_s"] = time.time() - started
    count = solved_count
    return SolveResult(
        frequencies_hz=solved_frequencies[:count],
        pressure_complex=pressure[:count],
        spl_db=spl[:count],
        impedance=impedance[:count],
        observation_angles_deg=angles,
        observation_planes=planes,
        config=config,
        mesh_info=mesh_info,
        timings=timings,
        solver_log=solver_log,
    )


_WARMUP_TETRAHEDRON = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
2 2 "warmup"
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 0.08 0.0 0.0
3 0.0 0.08 0.0
4 0.0 0.0 0.08
$EndNodes
$Elements
4
1 2 2 2 2 1 3 2
2 2 2 2 2 1 2 4
3 2 2 2 2 2 3 4
4 2 2 2 2 3 1 4
$EndElements
"""


def warm_up(
    *,
    beat_backend: str = "cpu",
    mode: str = "worker",
    julia_executable: str | None = None,
    julia_threads: str | int = "auto",
    julia_sysimage: str | Path | None = None,
    status_callback: Any = None,
) -> None:
    """Start (and optionally JIT-exercise) the persistent Julia worker.

    ``off`` does nothing; ``worker`` boots the worker so the first real solve
    skips Julia startup; ``tiny`` additionally solves one frequency on a
    4-triangle tetrahedron to trigger kernel compilation.
    """

    if mode not in {"off", "worker", "tiny"}:
        raise ValueError("warm-up mode must be off, worker, or tiny")
    if mode == "off":
        return
    julia = discover_julia(julia_executable)
    if julia is None:
        raise BeatUnavailableError(
            "No Julia executable was found. Install Julia or set HORNLAB_BEAT_JULIA."
        )
    from .runtime import default_project

    worker = get_worker(
        julia_executable=julia,
        solver_script=DEFAULT_SOLVER_SCRIPT,
        julia_threads=julia_threads,
        julia_project=default_project(beat_backend),
        julia_sysimage=None if julia_sysimage is None else Path(julia_sysimage),
    )
    worker.ensure_started(status_callback=status_callback)
    if mode != "tiny":
        return

    import tempfile

    from .config import ObservationConfig

    with tempfile.TemporaryDirectory(prefix="hornlab-beat-warmup-") as temp_dir:
        mesh_path = Path(temp_dir) / "warmup_tetrahedron.msh"
        mesh_path.write_text(_WARMUP_TETRAHEDRON, encoding="utf-8")
        config = SolveConfig(
            beat_backend=beat_backend,
            julia_executable=julia,
            julia_threads=julia_threads,
            julia_sysimage=julia_sysimage,
            observation=ObservationConfig(
                planes=["horizontal"],
                distance_m=1.0,
                angle_min_deg=0.0,
                angle_max_deg=90.0,
                angle_count=2,
            ),
        )
        solve_frequencies(mesh_path, [100.0], config, status_callback=status_callback)
