"""Procedurally generated demo dental arches (original work of this project, no third-party licence).

Teeth are squashed ellipsoids placed along an elliptical arch, FDI-labelled, in the same frame as the Teeth3DS data
(mm, -y labial, -z up, +x patient's right). They are NOT anatomical models: they exist so the demo, tests and a public
deployment can run without redistributing licensed scans. Real libraries are plugged in with DENTAL_LIBRARY_DIR.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .core import Arch, build_arch

# FDI position (1..8) -> (mesoDistal width mm, labio-lingual depth mm, crown height mm)
_DIMS = {1: (8.5, 7.0, 10.5), 2: (6.5, 6.0, 9.0), 3: (7.5, 8.0, 10.0), 4: (7.0, 8.5, 8.0), 5: (6.8, 9.0, 7.5),
         6: (10.0, 10.5, 7.0), 7: (9.5, 10.0, 6.5)}


def _tooth(width, depth, height, boxiness, rng, subdiv=2):
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    v = m.vertices
    v = np.sign(v) * np.abs(v) ** boxiness  # squash toward a rounded box
    v = v * np.array([width / 2, depth / 2, height / 2])
    v = v * (1 + rng.normal(0, 0.015, v.shape))  # slight surface irregularity
    return v, m.faces


def make_arch(seed: int, jaw: str = "upper", n_per_side: int = 6, name: str | None = None) -> Arch:
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.92, 1.08)
    wscale = {k: rng.uniform(0.9, 1.1) for k in _DIMS}
    A = 27.0 * scale * rng.uniform(0.95, 1.05)  # half arch width at the molars (mm)
    B = 24.0 * rng.uniform(0.9, 1.1)  # arch depth
    q1, q2 = (1, 2) if jaw == "upper" else (4, 3)
    # arc-length parametrisation of the half ellipse x = A sin t, y = -B cos t
    t = np.linspace(0, np.pi * 0.62, 400)
    pts = np.c_[A * np.sin(t), -B * np.cos(t)]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    parts = {}
    for side, quad in ((+1, q1), (-1, q2)):
        pos = -0.6  # small overlap at the midline
        for k in range(1, n_per_side + 1):
            w, d, h = _DIMS[k]
            w *= wscale[k]
            centre = pos + w / 2
            pos += w * rng.uniform(0.98, 1.0)
            i = int(np.clip(np.searchsorted(s, centre), 1, len(s) - 1))
            x, y = pts[i]
            tang = np.arctan2(pts[i][0] - pts[i - 1][0], -(pts[i][1] - pts[i - 1][1]))  # arch tangent angle about z
            v, f = _tooth(w, d, h, boxiness=0.55 if k <= 3 else 0.7, rng=rng)
            c, sn = np.cos(side * tang), np.sin(side * tang)
            v = v @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]).T
            z = -85.0 - 0.8 * (k - 1) + rng.normal(0, 0.3)  # gentle curve of Spee (molars higher, -z is up)
            parts[quad * 10 + k] = trimesh.Trimesh(v + np.array([side * x, y, z]), f, process=False)
    return build_arch(parts, name or f"proc_{jaw}_{seed:02d}", jaw)


def write_case_dir(arch: Arch, path):
    """Write per-tooth OBJ files in Person 2's layout (tooth_<FDI>.obj) so the OBJ loading path can be exercised."""
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    for fdi in sorted(arch.teeth):
        sel = arch.faces[arch.face_tooth == fdi]
        used, inv = np.unique(sel, return_inverse=True)
        trimesh.Trimesh(arch.vertices[used], inv.reshape(-1, 3), process=False).export(d / f"tooth_{fdi}.obj")
    return d
