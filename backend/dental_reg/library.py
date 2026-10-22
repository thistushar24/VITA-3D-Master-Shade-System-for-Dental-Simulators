"""Model library (compact .npz per arch) + catalogue with geometric features used for matching.

Library file: vertices (float32, mm), faces (int32), face_tooth (int16 FDI), teeth_ids, teeth_centroids (from the
ORIGINAL full-resolution meshes, mm). Original OBJ assets are never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import core
from .core import Arch, InputError

SOURCE = "Teeth3DS (processed by Person 2)"
LICENSE = "CC BY-NC-ND 4.0 (Teeth3DS): derivatives must NOT be redistributed - keep private"


def save_arch_npz(arch: Arch, path, centroids: dict | None = None):
    ids = sorted(arch.teeth)
    cen = centroids or arch.teeth
    np.savez_compressed(path, vertices=arch.vertices.astype(np.float32), faces=arch.faces.astype(np.int32),
                        face_tooth=arch.face_tooth.astype(np.int16), teeth_ids=np.array(ids, np.int16),
                        teeth_centroids=np.array([cen[i] for i in ids], np.float32), jaw=arch.jaw, name=arch.name)


def load_arch_npz(path) -> Arch:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"Library model missing: {p.name}. Check DENTAL_LIBRARY_DIR / run backend/build_library.py")
    try:
        z = np.load(p, allow_pickle=False)
        teeth = {int(i): c.astype(float) for i, c in zip(z["teeth_ids"], z["teeth_centroids"])}
        a = Arch(str(z["name"]), str(z["jaw"]), z["vertices"].astype(float), z["faces"].astype(int),
                 z["face_tooth"].astype(int), teeth)
    except Exception as e:
        raise InputError(f"Library model {p.name} unreadable: {e}") from e
    core.validate_mesh(__import__("trimesh").Trimesh(a.vertices, a.faces, process=False), p.name)
    return a


def model_features(a: Arch) -> dict:
    """Geometric descriptors in the arch frame (mm). Tooth-width features need the per-tooth labels."""
    V = a.vertices
    bb = [V.min(0).tolist(), V.max(0).tolist()]
    ext = np.ptp(V, axis=0)
    f = {"bbox_min": bb[0], "bbox_max": bb[1], "extent_mm": ext.tolist(), "n_vertices": int(len(V)),
         "n_faces": int(len(a.faces))}
    t = a.teeth
    if 13 in t and 23 in t:
        f["canine_distance_mm"] = float(np.linalg.norm(t[13] - t[23]))
    for fdi in (11, 21):
        if fdi in t:
            vs = V[np.unique(a.faces[a.face_tooth == fdi])]
            f[f"incisor_{fdi}_width_mm"] = float(np.ptp(vs[:, 0]))
    return f


def classify(a: Arch) -> str:
    """complete_arch (>=12 teeth incl. both canines + central incisors), partial_arch, single_mesh."""
    if not a.has_teeth:
        return "single_mesh"
    need = {11, 21, 13, 23} if a.jaw == "upper" else {31, 41, 33, 43}
    return "complete_arch" if need <= set(a.teeth) and len(a.teeth) >= 12 else "partial_arch"


def build_entry(a: Arch, model_id: str, file: str, shade=None) -> dict:
    return {"id": model_id, "file": file, "jaw": a.jaw, "classification": classify(a),
            "teeth": sorted(a.teeth), "tooth_count": len(a.teeth), "shade": shade,
            "source": SOURCE, "license": LICENSE, "preprocessing": "decimated, per-tooth labelled, npz",
            "features": model_features(a)}


def load_catalogue(lib_dir) -> list[dict]:
    p = Path(lib_dir) / "catalogue.json"
    if not p.is_file():
        raise InputError(f"Model catalogue not found at {p}. Run backend/build_library.py or set DENTAL_LIBRARY_DIR.")
    return json.loads(p.read_text())["models"]
