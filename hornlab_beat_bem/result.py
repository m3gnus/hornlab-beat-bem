from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ._constants import REFERENCE_PRESSURE
from .config import SolveConfig

#: Amplitudes are floored here before any dB conversion, so a silent sample
#: stays finite, the mapping stays monotonic near zero and ``log10`` never
#: sees 0. Equal to -120 dB re 20 uPa, the same floor hornlab-metal-bem
#: applies when it builds its directivity.
_DIRECTIVITY_FLOOR_PA = REFERENCE_PRESSURE * 10.0 ** (-120.0 / 20.0)


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

    # (F, P, N) — absolute SPL in dB re 20 µPa. ``directivity_db`` is the
    # on-axis-normalized view of the same field; the two are different
    # quantities and only one of them is what hornlab-metal-bem calls
    # directivity.
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

    #: True when the solver stopped early because cancellation was requested
    #: (``on_frequency_result`` returning ``False``, or a cancel written to
    #: the job directory). A cancelled sweep returns the frequencies it did
    #: solve; an *uncancelled* sweep that returns fewer is an error and
    #: raises instead, so this flag is the only way a short result appears.
    cancelled: bool = False

    #: How many frequencies the sweep asked the solver for, so a partial
    #: result describes itself. ``None`` when the producer did not say --
    #: results built by hand in tests and by older callers.
    requested_frequency_count: int | None = None

    @property
    def is_partial(self) -> bool:
        """True when fewer frequencies came back than were requested."""

        if self.requested_frequency_count is None:
            return False
        return int(np.asarray(self.frequencies_hz).size) < int(
            self.requested_frequency_count
        )

    @property
    def directivity_reference_index(self) -> int:
        """Index of the angle the normalized directivity is measured against.

        The first sample of smallest ``|angle|`` -- on axis for the ordinary
        grid, which ``ObservationConfig`` already requires to span 0 degrees.
        "First" makes a symmetric grid that straddles but misses zero (say
        -10, +10) resolve deterministically rather than by float noise.
        """

        angles = np.asarray(self.observation_angles_deg, dtype=np.float64)
        if angles.size == 0:
            raise ValueError(
                "directivity needs at least one observation angle to "
                "normalize against"
            )
        return int(np.argmin(np.abs(angles)))

    @property
    def directivity_reference_deg(self) -> float:
        """The angle ``directivity_db`` is normalized against, in degrees."""

        angles = np.asarray(self.observation_angles_deg, dtype=np.float64)
        return float(angles[self.directivity_reference_index])

    @property
    def directivity_db(self) -> NDArray[np.float64]:
        """(F, P, N) directivity in dB, reference angle = 0 dB.

        This is hornlab-metal-bem's ``directivity_db``, computed the same way
        and to the same reference, so a consumer written against that package
        reads the same numbers here. It is **not** absolute SPL: ``spl_db``
        is, and the two differ by the reference sample's level -- roughly
        94 dB for a 1 Pa reference, which is what this property returning
        ``spl_db`` used to hand a drop-in consumer.

        Normalization is per frequency and per plane, taken from
        ``pressure_complex`` rather than from ``spl_db`` so the floor applies
        before the subtraction and a null reference cannot produce
        ``inf - inf``. A reference quieter than the -120 dB floor therefore
        does not blow up; it lifts the whole cut by however far the floor sits
        above it, which is visible as an implausibly loud directivity rather
        than as ``nan``.
        """

        pressure = np.asarray(self.pressure_complex)
        if pressure.ndim != 3:
            raise ValueError(
                "pressure_complex must be (F, P, N) to normalize directivity; "
                f"got shape {pressure.shape}"
            )
        reference = self.directivity_reference_index
        if pressure.shape[2] != np.asarray(self.observation_angles_deg).size:
            raise ValueError(
                "pressure_complex angle axis does not match "
                f"observation_angles_deg: {pressure.shape[2]} != "
                f"{np.asarray(self.observation_angles_deg).size}"
            )
        amplitudes = np.maximum(np.abs(pressure), _DIRECTIVITY_FLOOR_PA)
        spl_raw = 20.0 * np.log10(amplitudes / REFERENCE_PRESSURE)
        return spl_raw - spl_raw[:, :, reference][:, :, None]

    @property
    def spl_norm_db(self) -> NDArray[np.float64]:
        """hornlab-metal-bem-compatible alias for ``directivity_db``."""

        return self.directivity_db
