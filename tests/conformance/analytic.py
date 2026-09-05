"""Closed forms and the meshes they are exact on.

The package-level counterpart of ``julia/scripts/validate_analytic_exterior.jl``.
That script fixes the *engine's* absolute answer; nothing in it can see a
convention defect introduced above the Julia boundary -- a conjugated
``_wire_rows`` unpacking, an acceleration rescale written ``1/(+i*omega)``, or a
sign flip when the cuts are stacked into ``(F, P, N)``. Those live in
``sweep.py`` and are exactly what a package-level analytic control catches.

Everything here is written in the conventions the package documents:
:math:`e^{-i\\omega t}` time dependence, outgoing waves :math:`e^{+ikr}`, and a
**unit normal acceleration** drive.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def icosphere(radius: float, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic icosphere vertices and outward-oriented triangles.

    Midpoint subdivision of an icosahedron with every vertex projected back
    onto the sphere, matching the generator the Julia analytic gate uses so the
    two report discretisation errors on comparable geometry. Windings are fixed
    against the outward radial direction rather than trusted: the body is
    star-shaped about its centre, so ``dot(normal, centroid)`` has an
    unambiguous sign.
    """

    if subdivisions < 0:
        raise ValueError("icosphere subdivision level must be >= 0")
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        (-1.0, phi, 0.0), (1.0, phi, 0.0), (-1.0, -phi, 0.0), (1.0, -phi, 0.0),
        (0.0, -1.0, phi), (0.0, 1.0, phi), (0.0, -1.0, -phi), (0.0, 1.0, -phi),
        (phi, 0.0, -1.0), (phi, 0.0, 1.0), (-phi, 0.0, -1.0), (-phi, 0.0, 1.0),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    points = [np.asarray(vertex, dtype=np.float64) for vertex in vertices]
    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            existing = midpoints.get(key)
            if existing is not None:
                return existing
            points.append((points[a] + points[b]) / 2.0)
            midpoints[key] = len(points) - 1
            return len(points) - 1

        refined: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = refined

    coordinates = np.stack(
        [radius * point / float(np.linalg.norm(point)) for point in points]
    )
    oriented = np.empty((len(faces), 3), dtype=np.int64)
    for index, (a, b, c) in enumerate(faces):
        v1, v2, v3 = coordinates[a], coordinates[b], coordinates[c]
        outward = (v1 + v2 + v3) / 3.0
        if float(np.dot(np.cross(v2 - v1, v3 - v1), outward)) >= 0.0:
            oriented[index] = (a, b, c)
        else:
            oriented[index] = (a, c, b)
    return coordinates, oriented


def write_gmsh22(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    physical_tag: int,
    physical_name: str = "source",
) -> Path:
    """Write a triangle surface as the Gmsh 2.2 ASCII the vendored loader reads."""

    destination = Path(path)
    lines = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        "1",
        f'2 {int(physical_tag)} "{physical_name}"',
        "$EndPhysicalNames",
        "$Nodes",
        str(int(vertices.shape[0])),
    ]
    for index, (x, y, z) in enumerate(vertices, start=1):
        lines.append(f"{index} {x:.17e} {y:.17e} {z:.17e}")
    lines.append("$EndNodes")
    lines.append("$Elements")
    lines.append(str(int(faces.shape[0])))
    tag = int(physical_tag)
    for index, (a, b, c) in enumerate(faces, start=1):
        lines.append(f"{index} 2 2 {tag} {tag} {a + 1} {b + 1} {c + 1}")
    lines.append("$EndElements")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def icosphere_surface_area(radius: float, subdivisions: int) -> float:
    """The faceted area, which is below ``4*pi*a^2`` by a known deficit."""

    vertices, faces = icosphere(radius, subdivisions)
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1).sum())


def pulsating_sphere_pressure_per_acceleration(
    observation_radius_m: float,
    sphere_radius_m: float,
    frequency_hz: float,
    *,
    air_density: float,
    sound_speed: float,
) -> complex:
    r"""Free-field pressure of a breathing sphere, per unit normal acceleration.

    For :math:`p = A e^{ikr}/r` the Euler relation
    :math:`u = \nabla p / (i\omega\rho)` and the surface condition
    :math:`u_r(a) = v` give

    .. math::
        p(r) = \rho c v \frac{a}{r} \frac{ika}{ika - 1} e^{ik(r-a)}

    which is the form the Julia analytic gate uses. This package reports every
    field per unit normal **acceleration**, i.e. rescaled by
    :math:`1/(-i\omega)`, and that rescale collapses the expression to

    .. math::
        p_a(r) = \frac{\rho a^2}{r}\frac{e^{ik(r-a)}}{1 - ika}

    whose :math:`k \to 0` limit is the incompressible :math:`\rho a^2 / r`,
    real and positive. The phase is the whole point of this function: under the
    opposite time convention every value here is its complex conjugate, and at
    :math:`k(r-a) \approx 53` rad that is not a subtle difference.
    """

    k = 2.0 * math.pi * frequency_hz / sound_speed
    a = sphere_radius_m
    r = observation_radius_m
    return (
        air_density
        * a
        * a
        / r
        * complex(np.exp(1j * k * (r - a)))
        / (1.0 - 1j * k * a)
    )


def level_and_phase_error(
    measured: np.ndarray, reference: complex | np.ndarray
) -> dict[str, float]:
    """Worst absolute level (dB) and phase (deg) error of ``measured``.

    Phase is compared as the argument of ``measured / reference``, so it wraps
    correctly and does not care how many multiples of 2*pi the propagation
    distance carries.
    """

    measured = np.asarray(measured, dtype=np.complex128).ravel()
    reference_array = np.asarray(reference, dtype=np.complex128).ravel()
    if reference_array.size not in (1, measured.size):
        raise ValueError(
            "reference must be a scalar or match the measured field: "
            f"{reference_array.size} against {measured.size}"
        )
    ratio = measured / np.broadcast_to(reference_array, measured.shape)
    level_db = 20.0 * np.log10(np.abs(ratio))
    phase_deg = np.degrees(np.angle(ratio))
    return {
        "worst_level_error_db": float(np.max(np.abs(level_db))),
        "worst_phase_error_deg": float(np.max(np.abs(phase_deg))),
        "rms_level_error_db": float(np.sqrt(np.mean(level_db**2))),
        "rms_phase_error_deg": float(np.sqrt(np.mean(phase_deg**2))),
    }
