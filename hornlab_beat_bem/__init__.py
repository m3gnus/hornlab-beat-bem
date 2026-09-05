"""BEAT Engine (boundary-lab) Julia BEM solver backend for HornLab.

Wraps the vendored BEAT Engine Burton-Miller solver (GPL-3.0, from
boundary-lab) behind the same native-result surface as hornlab-bempp-bem, so
WG's result mapping consumes it unmodified. The CPU backend is internal
scaffolding and CI coverage; the CUDA and ROCm backends are the product.
"""

from __future__ import annotations

from .capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    SOLVE_MODES,
    backend_capabilities,
    capability_report,
)
from .config import (
    BEAT_BACKENDS,
    BEAT_CPU,
    BEAT_CUDA,
    BEAT_METAL,
    BEAT_ROCM,
    GROUND_PLANE_AXES,
    GROUND_PLANE_AXIS_INDEX,
    GROUND_PLANE_TOLERANCE_M,
    GroundPlane,
    ObservationConfig,
    ObservationFrame,
    SolveConfig,
    beat_image_mode,
    beat_symmetry_mode,
    ground_plane_enabled,
    reject_unrepresentable_observation_origin,
    reject_unsupported_ground_plane,
    reject_unsupported_native_symmetry,
)
from .mesh import read_gmsh22_info
from .result import MeshInfo, SolveResult
from .runtime import (
    FORCE_CPU_ENV_VAR,
    JULIA_ENV_VAR,
    beat_engine_status,
    discover_julia,
    package_version,
    probe_gpu_functional_cache_clear,
)
from .submission import SubmissionClosed
from .sweep import (
    BeatUnavailableError,
    generated_frequency_grid,
    solve,
    solve_frequencies,
    warm_up,
)
from .worker import detach_workers, shutdown_workers

__all__ = [
    "BEAT_BACKENDS",
    "BEAT_CPU",
    "BEAT_CUDA",
    "BEAT_METAL",
    "BEAT_ROCM",
    "CAPABILITY_SCHEMA_VERSION",
    "FORCE_CPU_ENV_VAR",
    "GROUND_PLANE_AXES",
    "GROUND_PLANE_AXIS_INDEX",
    "GROUND_PLANE_TOLERANCE_M",
    "GroundPlane",
    "JULIA_ENV_VAR",
    "BeatUnavailableError",
    "MeshInfo",
    "ObservationConfig",
    "ObservationFrame",
    "REQUEST_SCHEMA_VERSION",
    "SOLVE_MODES",
    "SolveConfig",
    "SolveResult",
    "SubmissionClosed",
    "backend_capabilities",
    "beat_engine_status",
    "beat_image_mode",
    "beat_symmetry_mode",
    "capability_report",
    "detach_workers",
    "discover_julia",
    "ground_plane_enabled",
    "generated_frequency_grid",
    "package_version",
    "probe_gpu_functional_cache_clear",
    "read_gmsh22_info",
    "reject_unrepresentable_observation_origin",
    "reject_unsupported_ground_plane",
    "reject_unsupported_native_symmetry",
    "shutdown_workers",
    "solve",
    "solve_frequencies",
    "warm_up",
]
