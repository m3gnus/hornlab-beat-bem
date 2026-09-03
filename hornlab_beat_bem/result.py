from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .config import SolveConfig


@dataclass
class MeshInfo:
    n_vertices: int
    n_triangles: int
    physical_tag_areas_m2: dict[int, float]


@dataclass
class SolveResult:
    r"""BEAT Engine solve output in HornLab's shared native-result shape.

    Array dimensions use ``F`` for frequency count, ``P`` for observation
    plane count, and ``N`` for angles per plane. Complex values use the
    solver's :math:`e^{-i\omega t}` phase convention (outgoing waves
    :math:`e^{+ikr}`), the same convention as hornlab-metal-bem and
    hornlab-bempp-bem, and are reported for **unit normal acceleration** of
    the source tag (the Julia solver's 1 m/s velocity basis is rescaled by
    :math:`1/(-i\omega)` before it reaches these fields).
    """

    frequencies_hz: NDArray[np.float64]

    # (F, P, N) — complex pressure at every observation point
    pressure_complex: NDArray[np.complex128]

    # (F, P, N) — absolute SPL in dB re 20 µPa
    spl_db: NDArray[np.float64]

    # (F,) — raw area-weighted average pressure on the source tag under unit
    # acceleration. Follows hornlab-metal/bempp-bem; not normalized to rho*c.
    impedance: NDArray[np.complex128]

    observation_angles_deg: NDArray[np.float64]
    observation_planes: list[str]

    config: SolveConfig
    mesh_info: MeshInfo
    timings: dict[str, float] = field(default_factory=dict)
    solver_log: list[dict] = field(default_factory=list)

    # Populated whenever ``ObservationConfig.sphere_grid`` is set: the solver
    # samples a theta-major grid and the package rebuilds the exact float64
    # axes from the request, because the solver echoes them in Float32 radians
    # and DI integration checks the endpoints. ``None`` when no grid was asked
    # for. (F, n_theta*n_phi) and (n_theta*n_phi,).
    sphere_pressure_complex: NDArray[np.complex128] | None = None
    sphere_theta_deg: NDArray[np.float64] | None = None
    sphere_phi_deg: NDArray[np.float64] | None = None

    # Boundary Cauchy datum, populated when ``SolveConfig.surface_traces`` is
    # set: (F, n_vertices) P1 pressure and (F, n_faces) DP0 normal derivative,
    # in the same unit-acceleration convention as ``pressure_complex``. Names
    # match hornlab-metal-bem and hornlab-bempp-bem so HornLab's field-trace
    # artifact builder consumes all three identically. Under a reduced-mesh
    # symmetry solve these cover the fundamental domain, not the full body.
    surface_pressure_complex: NDArray[np.complex128] | None = None
    surface_neumann_complex: NDArray[np.complex128] | None = None

    @property
    def directivity_db(self) -> NDArray[np.float64]:
        """hornlab_metal_bem-compatible name for spl_db."""
        return self.spl_db
