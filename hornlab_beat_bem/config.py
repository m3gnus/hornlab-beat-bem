from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from ._constants import AIR_DENSITY, SPEED_OF_SOUND

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


BEAT_CPU = "cpu"
BEAT_CUDA = "cuda"
BEAT_ROCM = "rocm"
BEAT_BACKENDS = (BEAT_CPU, BEAT_CUDA, BEAT_ROCM)

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
        unknown = set(self.planes) - {"horizontal", "vertical"}
        if unknown:
            names = ", ".join(sorted(repr(name) for name in unknown))
            raise ValueError(
                f"planes contain unsupported name(s): {names}; the BEAT Engine "
                "solver evaluates fixed horizontal (x-z) and vertical (y-z) cuts"
            )

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
    # unit-amplitude normal-motion source tag are supported; see class docs.
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
    singular_order: int = 4
    #: None = solver default ("wavelength" on cpu, "fixed" on accelerators).
    regular_quadrature_mode: Literal["fixed", "wavelength"] | None = None

    # Execution backend
    beat_backend: Literal["cpu", "cuda", "rocm"] = "cpu"
    julia_executable: str | None = None
    julia_project: str | Path | None = None
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
            raise ValueError("beat_backend must be 'cpu', 'cuda', or 'rocm'")
        if self.source_motion != "normal":
            raise NotImplementedError(
                "hornlab-beat-bem only supports source_motion='normal'; the "
                "BEAT Engine solver has no axial piston projection"
            )
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
        if not _is_integral_value(self.singular_order) or not 1 <= int(self.singular_order) <= 4:
            raise ValueError("singular_order must be between 1 and 4")
        if self.regular_quadrature_mode not in {None, "fixed", "wavelength"}:
            raise ValueError("regular_quadrature_mode must be None, 'fixed', or 'wavelength'")

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
