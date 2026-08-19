"""BEAT Engine (boundary-lab) Julia BEM solver backend for HornLab.

Wraps the vendored BEAT Engine Burton-Miller solver (GPL-3.0, from
boundary-lab) behind the same native-result surface as hornlab-bempp-bem, so
WG's result mapping consumes it unmodified. The CPU backend is internal
scaffolding and CI coverage; the CUDA and ROCm backends are the product.
"""

from __future__ import annotations

from .config import (
    BEAT_BACKENDS,
    BEAT_CPU,
    BEAT_CUDA,
    BEAT_ROCM,
    ObservationConfig,
    ObservationFrame,
    SolveConfig,
    beat_symmetry_mode,
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
from .sweep import (
    BeatUnavailableError,
    generated_frequency_grid,
    solve,
    solve_frequencies,
    warm_up,
)
from .worker import shutdown_workers

__all__ = [
    "BEAT_BACKENDS",
    "BEAT_CPU",
    "BEAT_CUDA",
    "BEAT_ROCM",
    "FORCE_CPU_ENV_VAR",
    "JULIA_ENV_VAR",
    "BeatUnavailableError",
    "MeshInfo",
    "ObservationConfig",
    "ObservationFrame",
    "SolveConfig",
    "SolveResult",
    "beat_engine_status",
    "beat_symmetry_mode",
    "discover_julia",
    "generated_frequency_grid",
    "package_version",
    "probe_gpu_functional_cache_clear",
    "read_gmsh22_info",
    "reject_unsupported_native_symmetry",
    "shutdown_workers",
    "solve",
    "solve_frequencies",
    "warm_up",
]
