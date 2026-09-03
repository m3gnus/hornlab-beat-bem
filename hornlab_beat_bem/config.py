from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ._constants import AIR_DENSITY, SPEED_OF_SOUND

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


BEAT_CPU = "cpu"
BEAT_CUDA = "cuda"
BEAT_ROCM = "rocm"
BEAT_METAL = "metal"
BEAT_BACKENDS = (BEAT_CPU, BEAT_CUDA, BEAT_ROCM, BEAT_METAL)

NativeSymmetryPlane = Literal["yz", "xz", "xy", "yz+xz"]

#: WG native-symmetry plane -> BEAT Engine symmetry mode. BEAT mirrors across
#: x=0 (``x``) or across both x=0 and y=0 (``xy``); it has no y-only mirror, so
#: WG's ``xz`` half-domain (quadrants "12") is not representable without an
#: axis swap that would also swap the observation cuts.
_SYMMETRY_PLANE_TO_BEAT_MODE = {None: "off", "yz": "x", "yz+xz": "xy"}


def _is_integral_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return isfinite(float(value)) and int(value) == value
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass
class ObservationFrame:
    """Explicit observation basis, matching hornlab-bempp-bem's frame shape.

    ``axis`` must be (numerically) +z, ``u`` +x, and ``v`` +y: the vendored
    BEAT solver computes its polar cuts in the mesh's own global frame around
    the coordinate origin, so the only frame freedom this package has is a
    rigid translation (applied as a mesh translation so ``origin`` lands on
    the solver's coordinate origin). ``sweep.solve_frequencies`` validates
    the orientation and rejects tilted frames rather than silently measuring
    the wrong arc.
    """

    axis: NDArray[np.float64]
    origin: NDArray[np.float64]
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    mouth_center: NDArray[np.float64] | None = None
    source_center: NDArray[np.float64] | None = None


@dataclass
class ObservationConfig:
    planes: list[str] = field(default_factory=lambda: ["horizontal", "vertical"])
    distance_m: float = 2.0
    angle_min_deg: float = 0.0
    angle_max_deg: float = 180.0
    angle_count: int = 37
    origin: Literal["mouth", "throat"] = "mouth"

    #: Diagonal-cut rotation from the horizontal toward the vertical plane.
    inclination_deg: float = 45.0

    #: Frame-relative spherical field grid as (n_theta, n_phi), theta-major,
    #: theta in [0, sphere_theta_max_deg] inclusive and phi in [0, 360)
    #: without a wrap column -- the layout HornLab's balloon/DI mapping needs.
    sphere_grid: tuple[int, int] | None = None
    sphere_theta_max_deg: float = 180.0

    def __post_init__(self) -> None:
        try:
            distance_m = float(self.distance_m)
        except (TypeError, ValueError, OverflowError):
            distance_m = float("nan")
        if not isfinite(distance_m) or distance_m <= 0.0:
            raise ValueError("distance_m must be finite and greater than zero")
        self.distance_m = distance_m

        for field_name in ("angle_min_deg", "angle_max_deg"):
            try:
                value = float(getattr(self, field_name))
            except (TypeError, ValueError, OverflowError):
                value = float("nan")
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            setattr(self, field_name, value)
        if self.angle_max_deg < self.angle_min_deg:
            raise ValueError("angle_max_deg must be >= angle_min_deg")
        if not (self.angle_min_deg <= 0.0 <= self.angle_max_deg):
            # The vendored solver anchors its normalisation and cut layout on
            # the 0-degree sample and enforces this itself; failing here keeps
            # the refusal before any Julia process is started.
            raise ValueError(
                "hornlab-beat-bem polar ranges must include 0 degrees"
            )
        if not (-180.0 <= self.angle_min_deg and self.angle_max_deg <= 180.0):
            raise ValueError("polar angles must stay within [-180, 180] degrees")

        if not _is_integral_value(self.angle_count) or self.angle_count < 1:
            raise ValueError("angle_count must be a positive integer")
        self.angle_count = int(self.angle_count)
        if self.angle_count == 1 and self.angle_min_deg != self.angle_max_deg:
            raise ValueError("angle_count 1 requires angle_min_deg == angle_max_deg")

        if self.origin not in ("mouth", "throat"):
            raise ValueError("origin must be 'mouth' or 'throat'")

        if not isinstance(self.planes, list) or not self.planes:
            raise ValueError("planes must be a non-empty list of plane names")
        if len(set(self.planes)) != len(self.planes):
            raise ValueError("planes must not contain duplicates")
        unknown = set(self.planes) - {"horizontal", "vertical", "diagonal"}
        if unknown:
            names = ", ".join(sorted(repr(name) for name in unknown))
            raise ValueError(f"planes contain unknown name(s): {names}")

        try:
            inclination_deg = float(self.inclination_deg)
        except (TypeError, ValueError, OverflowError):
            inclination_deg = float("nan")
        if not isfinite(inclination_deg):
            raise ValueError("inclination_deg must be finite")
        self.inclination_deg = inclination_deg

        if self.sphere_grid is not None:
            try:
                grid = tuple(self.sphere_grid)
            except TypeError:
                raise ValueError("sphere_grid must be an iterable (n_theta, n_phi)") from None
            if len(grid) != 2 or not all(_is_integral_value(value) for value in grid):
                raise ValueError("sphere_grid must be two integers (n_theta, n_phi)")
            n_theta, n_phi = int(grid[0]), int(grid[1])
            if n_theta < 2:
                raise ValueError("sphere_grid n_theta must be at least 2")
            if n_phi < 3:
                raise ValueError("sphere_grid n_phi must be at least 3")
            if n_theta * n_phi > 100_000:
                raise ValueError("sphere_grid is too dense (n_theta*n_phi > 100000)")
            self.sphere_grid = (n_theta, n_phi)
        try:
            sphere_theta_max_deg = float(self.sphere_theta_max_deg)
        except (TypeError, ValueError, OverflowError):
            sphere_theta_max_deg = float("nan")
        if not (0.0 < sphere_theta_max_deg <= 180.0):
            raise ValueError("sphere_theta_max_deg must be in (0, 180]")
        self.sphere_theta_max_deg = sphere_theta_max_deg

    @property
    def step_deg(self) -> float:
        if self.angle_count == 1:
            return 1.0
        return (self.angle_max_deg - self.angle_min_deg) / (self.angle_count - 1)


@dataclass
class SolveConfig:
    """One BEAT Engine sweep.

    Result convention: although the Julia solver drives its boundary condition
    as ``q = i*rho*omega*v_n`` with a 1 m/s normal-velocity basis, every array
    this package returns is converted to the **unit normal acceleration**
    convention used by hornlab-metal-bem and hornlab-bempp-bem
    (``p_accel = p_vel / (-i*omega)``, e^{-i omega t} time convention,
    outgoing waves e^{+ikr}), so WG's result mapping applies unmodified.
    """

    # Frequency sweep (summary; solve_frequencies() takes the explicit list)
    freq_min_hz: float = 500.0
    freq_max_hz: float = 20_000.0
    freq_count: int = 40
    freq_spacing: Literal["log", "linear"] = "log"

    # Boundary condition. Only the acceleration output convention and a single
    # unit-amplitude source tag are supported; see class docs. "normal" drives
    # a uniformly breathing cap; "axial" a rigid piston along the +z axis
    # (per-face velocity scaled by n_hat . z), mirroring the other backends.
    velocity_sources: dict[int, float] = field(default_factory=lambda: {2: 1.0})
    source_motion: str = "normal"

    # Observation
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    #: Optional explicit frame; must be axis-aligned (+z axis). When set, the
    #: mesh is translated so frame.origin coincides with the solver origin.
    frame_override: ObservationFrame | None = None

    #: Reduced-mesh mirror plane(s), in WG's quadrant naming. ``yz`` and
    #: ``yz+xz`` map onto BEAT's ``x``/``xy`` symmetry; ``xz`` and legacy
    #: ``xy`` are rejected by ``reject_unsupported_native_symmetry``.
    native_symmetry_plane: NativeSymmetryPlane | None = None

    # Mesh scale applied on load (1.0 = mesh already in metres).
    mesh_scale: float = 1.0

    # Physics constants passed into the Julia request so the solve runs at the
    # same air the rest of HornLab reports against.
    air_density: float = AIR_DENSITY
    sound_speed: float = SPEED_OF_SOUND

    # Quadrature (BEAT-native orders: Gauss point counts 1/3/6 for 1/2/4).
    quadrature_order: int = 4
    #: Duffy 1-D Gauss order for coincident and edge/vertex-adjacent pairs.
    #: Cost per singular pair grows as order^4. Orders above 4 need a solver
    #: whose ``gauss_rule_1d`` computes nodes by Golub-Welsch rather than the
    #: hard-coded 1..4 table -- true from boundary-lab dev c7bf772 onward --
    #: and they require ``solve_precision="double"``: measured on the ASRO
    #: quarter mesh, order 4 is already converged to 0.0016 dB rms in Float64,
    #: while in Float32 the extra Duffy points amplify cancellation faster
    #: than they cut quadrature error (0.0006 dB rms of single-precision noise
    #: at order 4, 0.031 dB at order 8). Raising it on the GPU path is a
    #: pessimisation, so it is refused rather than merely discouraged.
    singular_order: int = 4
    #: None = solver default ("wavelength" on cpu, "fixed" on accelerators).
    regular_quadrature_mode: Literal["fixed", "wavelength"] | None = None

    #: Near-singular correction. Disjoint element pairs closer than
    #: ``near_correction_cutoff`` times their combined circumradius are
    #: re-integrated with a tensor-product Gauss rule and the plain regular
    #: contribution is subtracted. Each pair's order follows the separation
    #: ratio (Bernstein-ellipse convergence), floored at 4 and capped at
    #: ``near_correction_order``. Under a symmetry mode it also covers
    #: mirror-image pairs, one cache per image transform. Off reproduces the
    #: uncorrected solve bit-for-bit. CPU and Metal only -- see the
    #: validation below for why CUDA and ROCm refuse it.
    near_correction: bool = False
    near_correction_cutoff: float = 2.0
    near_correction_order: int = 8

    #: Retain the boundary Cauchy datum (P1 pressure per vertex, DP0 normal
    #: derivative per face) for every solved frequency, so a caller can
    #: reconstruct field planes after the solve. Off by default because it is
    #: O(vertices + faces) complex numbers per frequency on the wire, against
    #: the polar cuts' few dozen.
    surface_traces: bool = False

    #: Assembly/solve element type. BEAT is a Float32 engine because that is
    #: what its GPU kernels are; the CPU path is type-generic, so "double"
    #: runs the identical solve in Float64. CPU backend only. Use it to tell
    #: single-precision noise apart from discretisation error when comparing
    #: against a Float64 reference solver.
    solve_precision: Literal["single", "double"] = "single"

    # Execution backend
    beat_backend: Literal["cpu", "cuda", "rocm", "metal"] = "cpu"
    julia_executable: str | None = None
    julia_project: str | Path | None = None
    #: "auto" resolves to the performance-core count, not os.cpu_count(); the
    #: Metal path hands this straight to BLAS, and a factorization thread on an
    #: efficiency core holds up the rest. An explicit count is used as given.
    julia_threads: str | int = "auto"
    julia_sysimage: str | Path | None = None
    persistent_worker: bool = True

    # Callbacks, matching hornlab-bempp-bem's seam:
    # progress_callback(freq_index, total, frequency_hz)
    progress_callback: Callable[[int, int, float], None] | None = None
    # on_frequency_result(freq_index, frequency_hz, log_entry) -> bool
    # Returning exactly False cancels the sweep (partial result is built).
    on_frequency_result: Callable[[int, float, dict], bool] | None = None

    def __post_init__(self) -> None:
        if self.freq_spacing not in {"log", "linear"}:
            raise ValueError("freq_spacing must be 'log' or 'linear'")
        if not _is_integral_value(self.freq_count) or self.freq_count < 1:
            raise ValueError("freq_count must be at least 1")
        self.freq_count = int(self.freq_count)
        for field_name in ("freq_min_hz", "freq_max_hz", "mesh_scale", "air_density", "sound_speed"):
            value = getattr(self, field_name)
            try:
                valid = isfinite(value) and value > 0.0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"{field_name} must be finite and greater than zero")
        if self.freq_min_hz > self.freq_max_hz:
            raise ValueError("freq_min_hz must not exceed freq_max_hz")
        if self.beat_backend not in BEAT_BACKENDS:
            raise ValueError(
                "beat_backend must be one of " + ", ".join(repr(name) for name in BEAT_BACKENDS)
            )
        if self.source_motion not in {"normal", "axial"}:
            raise ValueError("source_motion must be 'normal' or 'axial'")
        if not isinstance(self.velocity_sources, dict) or len(self.velocity_sources) != 1:
            raise NotImplementedError(
                "hornlab-beat-bem supports exactly one velocity source tag"
            )
        ((tag, amplitude),) = self.velocity_sources.items()
        if not _is_integral_value(tag):
            raise ValueError("velocity_sources tag must be an integer")
        if float(amplitude) != 1.0:
            raise NotImplementedError(
                "hornlab-beat-bem supports unit source amplitude only"
            )
        if self.native_symmetry_plane not in {None, "yz", "xz", "xy", "yz+xz"}:
            raise ValueError(
                "native_symmetry_plane must be None, 'yz', 'xz', 'xy', or 'yz+xz'"
            )
        if not _is_integral_value(self.quadrature_order) or int(self.quadrature_order) < 1:
            raise ValueError("quadrature_order must be a positive integer")
        if not _is_integral_value(self.singular_order) or not 1 <= int(self.singular_order) <= 12:
            raise ValueError("singular_order must be between 1 and 12")
        if self.regular_quadrature_mode not in {None, "fixed", "wavelength"}:
            raise ValueError("regular_quadrature_mode must be None, 'fixed', or 'wavelength'")
        if not isinstance(self.near_correction, bool):
            raise ValueError("near_correction must be a bool")
        try:
            cutoff = float(self.near_correction_cutoff)
        except (TypeError, ValueError, OverflowError):
            cutoff = float("nan")
        if not isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("near_correction_cutoff must be finite and greater than zero")
        self.near_correction_cutoff = cutoff
        if not _is_integral_value(self.near_correction_order) or int(self.near_correction_order) < 4:
            raise ValueError("near_correction_order must be an integer >= 4")
        self.near_correction_order = int(self.near_correction_order)
        if self.near_correction and self.beat_backend in {BEAT_ROCM, BEAT_CUDA}:
            # ROCm's vendored assembly has no near-pair kernel at all, so
            # accepting the flag there would report a corrected solve that
            # never ran the correction.
            #
            # CUDA is refused for a narrower and more dangerous reason: it has
            # the kernel, but upstream's device path still takes a *single*
            # image-near cache. Under `yz+xz` that leaves two of the three
            # mirror transforms integrating a near-singular kernel with the
            # plain regular rule and says nothing about it -- a plausible wrong
            # number rather than a crash. The CPU assembly in this package
            # carries the multi-cache patch; the CUDA path deliberately does
            # not, because no machine available to this project has an NVIDIA
            # GPU, so the change could not be executed or measured. Lift this
            # once someone with the hardware ports and validates it.
            label = "ROCm" if self.beat_backend == BEAT_ROCM else "CUDA"
            raise NotImplementedError(
                f"near_correction is not implemented for the BEAT {label} backend"
            )
        if not isinstance(self.surface_traces, bool):
            raise ValueError("surface_traces must be a bool")
        if self.solve_precision not in {"single", "double"}:
            raise ValueError("solve_precision must be 'single' or 'double'")
        if self.solve_precision == "double" and self.beat_backend != BEAT_CPU:
            raise NotImplementedError(
                "solve_precision='double' is only available on the BEAT CPU backend"
            )
        if self.singular_order > 4 and self.solve_precision != "double":
            raise ValueError(
                "singular_order above 4 requires solve_precision='double'; in "
                "single precision the extra Duffy points add more cancellation "
                "noise than they remove quadrature error"
            )

    @property
    def source_tag(self) -> int:
        ((tag, _),) = self.velocity_sources.items()
        return int(tag)


def beat_symmetry_mode(plane: NativeSymmetryPlane | None) -> str:
    """Map a WG native-symmetry plane onto the BEAT solver's symmetry mode."""

    try:
        return _SYMMETRY_PLANE_TO_BEAT_MODE[plane]
    except KeyError:
        raise NotImplementedError(
            f"hornlab-beat-bem cannot represent native symmetry plane {plane!r}"
        ) from None


def reject_unsupported_native_symmetry(config: SolveConfig) -> None:
    """Reject symmetry modes the BEAT solver cannot express, before any solve.

    BEAT mirrors across x=0 and optionally y=0 with the reduced mesh in the
    positive fundamental domain. WG's ``yz`` half and ``yz+xz`` quarter map
    directly; a y-only ``xz`` half (quadrants "12") and legacy ``xy`` do not.
    """

    if config.native_symmetry_plane in {"xz", "xy"}:
        raise NotImplementedError(
            "hornlab-beat-bem native symmetry supports 'yz' half and 'yz+xz' "
            f"quarter domains; {config.native_symmetry_plane!r} is not "
            "representable by the BEAT Engine solver"
        )
