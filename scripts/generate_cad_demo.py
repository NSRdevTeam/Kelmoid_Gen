#!/usr/bin/env python3
"""
Genesis CAD demo generator
- Uses AdvancedCAD primitives to build a simple assembly
- Exports to outputs/demo_model.stl

This script will work if either:
- AdvancedCAD is imported into this repo at AdvancedCAD/src, or
- A sibling directory exists at ../AdvancedCAD/src
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import os
from typing import List, Tuple

# Resolve project paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Candidate locations for AdvancedCAD/src
ADV_IN_REPO = REPO_ROOT / "AdvancedCAD" / "src"
ADV_SIBLING = REPO_ROOT.parent / "AdvancedCAD" / "src"

for cand in [ADV_IN_REPO, ADV_SIBLING]:
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

try:
    from core.primitives import create_primitive, MeshData
    from core.file_io import export_mesh
except Exception as e:
    raise SystemExit(
        "Could not import AdvancedCAD core. Ensure AdvancedCAD is added as a subtree under 'AdvancedCAD/' or exists as a sibling directory '../AdvancedCAD/'.\n"
        f"Import error: {e}"
    )


def translate_mesh(mesh: MeshData, offset: Tuple[float, float, float]) -> MeshData:
    import numpy as np

    dx, dy, dz = offset
    translated = MeshData(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        normals=mesh.normals.copy() if mesh.normals is not None else None,
        colors=mesh.colors.copy() if mesh.colors is not None else None,
        uvs=mesh.uvs.copy() if mesh.uvs is not None else None,
    )
    translated.vertices += np.array([dx, dy, dz], dtype=translated.vertices.dtype)
    return translated


def combine_meshes(meshes: List[MeshData]) -> MeshData:
    import numpy as np

    if not meshes:
        raise ValueError("No meshes to combine")

    vertices_list = []
    faces_list = []
    normals_list = []

    vertex_offset = 0
    for m in meshes:
        v = m.vertices
        f = m.faces
        vertices_list.append(v)
        faces_list.append(f + vertex_offset)
        vertex_offset += v.shape[0]

        if m.normals is not None:
            normals_list.append(m.normals)

    vertices = np.vstack(vertices_list)
    faces = np.vstack(faces_list)
    normals = np.vstack(normals_list) if normals_list else None

    return MeshData(vertices=vertices, faces=faces, normals=normals)


def generate_demo_model() -> MeshData:
    # Base box
    box = create_primitive("cube", {"size": [40, 30, 10], "center": True})

    # Cylinder post translated to sit on top of the box
    cyl = create_primitive("cylinder", {"height": 20, "radius": 6, "center": True})
    # Move cylinder up by half box height + half cylinder height
    cyl = translate_mesh(cyl, (0, 0, 10/2 + 20/2))

    # Combine
    return combine_meshes([box, cyl])


def main() -> None:
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_model.stl"

    mesh = generate_demo_model()
    ok = export_mesh(mesh, str(out_path), binary=True)

    if not ok:
        raise SystemExit("Export failed")

    print(f"Exported demo model to {out_path}")


if __name__ == "__main__":
    main()
