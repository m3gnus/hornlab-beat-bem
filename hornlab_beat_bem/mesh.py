from __future__ import annotations

from pathlib import Path

import numpy as np

from .result import MeshInfo


def read_gmsh22_info(path: str | Path, *, scale: float = 1.0) -> MeshInfo:
    """Read the Gmsh 2.2 ASCII facts this package needs.

    Per-tag triangle areas (in metres, after ``scale``) let ``sweep`` turn the
    solver's throat force back into the area-weighted mean pressure HornLab's
    impedance contract expects. The parse mirrors the vendored Julia loader:
    triangles only, last three columns are the node ids.
    """

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        nodes_start = lines.index("$Nodes")
        elements_start = lines.index("$Elements")
        node_count = int(lines[nodes_start + 1])
        element_count = int(lines[elements_start + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"{path} is not a Gmsh 2.2 ASCII mesh with Nodes and Elements sections"
        ) from exc

    nodes: dict[int, np.ndarray] = {}
    for row in lines[nodes_start + 2 : nodes_start + 2 + node_count]:
        parts = row.split()
        if len(parts) >= 4:
            nodes[int(parts[0])] = np.asarray(parts[1:4], dtype=np.float64) * scale

    tag_areas: dict[int, float] = {}
    triangle_count = 0
    for row in lines[elements_start + 2 : elements_start + 2 + element_count]:
        parts = row.split()
        if len(parts) < 8 or int(parts[1]) != 2:
            continue
        physical_tag = int(parts[3])
        try:
            v1, v2, v3 = (nodes[int(value)] for value in parts[-3:])
        except KeyError:
            continue
        area = 0.5 * float(np.linalg.norm(np.cross(v2 - v1, v3 - v1)))
        tag_areas[physical_tag] = tag_areas.get(physical_tag, 0.0) + area
        triangle_count += 1

    return MeshInfo(
        n_vertices=len(nodes),
        n_triangles=triangle_count,
        physical_tag_areas_m2=tag_areas,
    )
