"""v1 demo API: Photo A/B session -> automatic model match -> align -> confirm -> edit versions -> Photo B placement.

Mounted by app.py under /api/v1. Pose/offsets/metrics/export reuse the existing /api/registrations machinery.
State: in-memory + a session.json under STORAGE (best effort; restart loses in-memory sessions).
"""
from __future__ import annotations

import json
import shutil
import time
import uuid

import numpy as np
from fastapi import APIRouter

from . import core, library, match, vision
from .core import InputError

SESSIONS: dict[str, dict] = {}


def make_router(*, images, storage, lib_dir, create_reg, auto_align, set_transform, get_reg) -> APIRouter:
    r = APIRouter(prefix="/api/v1")

    def sess(i):
        if i not in SESSIONS and isinstance(i, str) and i.isalnum():  # reload after a restart
            f = storage() / f"session_{i}" / "session.json"
            if f.is_file():
                SESSIONS[i] = json.loads(f.read_text())
        if i not in SESSIONS:
            raise KeyError(i)
        return SESSIONS[i]

    def persist(s):
        d = storage() / f"session_{s['id']}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.json").write_text(json.dumps({k: v for k, v in s.items() if k != "_masks"}, indent=1, default=str))

    def img(i):
        if i not in images:
            raise InputError(f"unknown image id {i!r}")
        return core.load_image(images[i]["path"])

    def view(s):
        return {"ok": True, **{k: v for k, v in s.items() if k != "_masks"}}

    def upper_mask(s):
        a = s["photo_a"]
        polys = s["analysis"]["mask_polygons"]
        if not polys:
            raise InputError("No tooth region yet: run analysis or draw an outline")
        return core.mask_from_polygons(polys, a["width"], a["height"])

    @r.get("/catalogue")
    def catalogue():
        cat = library.load_catalogue(lib_dir())
        return {"ok": True, "models": cat, "count": len(cat), "library_dir": str(lib_dir())}

    @r.post("/sessions")
    def create_session(body: dict):
        """body: {photo_a_id, photo_b_id?}. Runs automatic tooth-region segmentation on Photo A (editable result)."""
        ia = img(body.get("photo_a_id"))
        h, w = ia.shape[:2]
        s = {"id": uuid.uuid4().hex[:12], "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "photo_a": {"id": body["photo_a_id"], "width": w, "height": h},
             "photo_b": None, "analysis": {"mask_polygons": [], "canines": None, "notes": [], "confidence": "none"},
             "match": None, "registration_id": None, "model_id": None, "alignment_a": None,
             "versions": [], "photo_b_placement": None, "stage": "uploaded"}
        if body.get("photo_b_id"):
            ib = img(body["photo_b_id"])
            s["photo_b"] = {"id": body["photo_b_id"], "width": ib.shape[1], "height": ib.shape[0]}
        try:
            seg = vision.segment_teeth(ia)
            s["analysis"] = {"mask_polygons": vision.contours_from_mask(seg["mask_upper"]), "canines": seg["canines"],
                             "notes": seg["notes"], "confidence": seg["confidence"], "source": "automatic"}
        except InputError as e:
            s["analysis"]["notes"] = [str(e)]
        SESSIONS[s["id"]] = s
        persist(s)
        return view(s)

    @r.get("/sessions/{i}")
    def get_session(i: str):
        return view(sess(i))

    @r.put("/sessions/{i}/annotations")
    def set_annotations(i: str, body: dict):
        """User-edited canine points (image px: [[xl,yl],[xr,yr]]) and/or tooth outline polygons."""
        s = sess(i)
        w, h = s["photo_a"]["width"], s["photo_a"]["height"]
        if body.get("canines") is not None:
            s["analysis"]["canines"] = core.validate_points(body["canines"], w, h, "canines").tolist()
            if len(s["analysis"]["canines"]) != 2:
                raise InputError("exactly two canine points are required (patient's right = image left, then patient's left)")
        if body.get("mask_polygons") is not None:
            for p in body["mask_polygons"]:
                core.validate_points(p, w, h, "mask polygon")
            s["analysis"]["mask_polygons"] = body["mask_polygons"]
        s["analysis"]["source"] = "user-edited"
        persist(s)
        return view(s)

    def _register(s, model_id, fov):
        a = s["analysis"]
        ann = {"landmarks": [{"fdi": 13, "image_xy": a["canines"][0]}, {"fdi": 23, "image_xy": a["canines"][1]}]
               if a["canines"] else [], "mask_polygons": a["mask_polygons"]}
        reg = create_reg({"image_id": s["photo_a"]["id"], "model_id": model_id, "camera": {"fov_deg": fov}, "annotations": ann})
        try:
            reg = auto_align(reg["id"], {})
        except InputError as e:  # nothing to fit with: keep the initial pose, tell the user
            s["analysis"]["notes"].append(f"automatic placement skipped: {e}")
        s["registration_id"], s["model_id"], s["fov_deg"] = reg["id"], model_id, fov
        s["stage"] = "matched"
        return reg

    @r.post("/sessions/{i}/match")
    def run_match(i: str, body: dict | None = None):
        """Automatically rank the catalogue and select the best candidate (see match.py for the scoring function)."""
        s, body = sess(i), body or {}
        a = s["analysis"]
        if not a["canines"]:
            raise InputError("Canine reference points are needed: drag the two points onto the canines")
        fov = float(body.get("fov_deg", 40.0))
        res = match.rank_models(upper_mask(s), a["canines"], library.load_catalogue(lib_dir()), lib_dir(),
                                (s["photo_a"]["height"], s["photo_a"]["width"], 3), fov_deg=fov,
                                top_k=int(body.get("top_k", 6)))
        s["match"] = res
        reg = _register(s, f"lib:{res['selected']}", fov)
        persist(s)
        return {**view(s), "registration": reg}

    @r.post("/sessions/{i}/replace_model")
    def replace_model(i: str, body: dict):
        s = sess(i)
        ids = {e["id"] for e in library.load_catalogue(lib_dir())}
        if body.get("model_id") not in ids:
            raise InputError(f"model {body.get('model_id')!r} is not in the catalogue")
        reg = _register(s, f"lib:{body['model_id']}", s.get("fov_deg", 40.0))
        if s["match"]:
            s["match"]["user_replaced"] = body["model_id"]
        persist(s)
        return {**view(s), "registration": reg}

    @r.post("/sessions/{i}/confirm_alignment")
    def confirm(i: str, body: dict):
        """Save the Photo A alignment (pose) and create the editable working copy (version 1). Original asset untouched."""
        s = sess(i)
        if not s["registration_id"]:
            raise InputError("Run model matching first")
        reg = set_transform(s["registration_id"], {"pose": body["pose"]}) if body.get("pose") else get_reg(s["registration_id"])
        s["alignment_a"] = {"pose": reg["pose"], "camera": reg["camera"], "model_id": s["model_id"], "report": reg.get("report")}
        entry = next(e for e in library.load_catalogue(lib_dir()) if f"lib:{e['id']}" == s["model_id"])
        d = storage() / f"session_{s['id']}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lib_dir() / entry["file"], d / "working_model.npz")  # working copy
        s["versions"] = [{"n": 1, "note": "confirmed alignment (unedited working copy)", "tooth_offsets": {},
                          "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
        s["stage"] = "editing"
        persist(s)
        return view(s)

    @r.post("/sessions/{i}/versions")
    def save_version(i: str, body: dict):
        s = sess(i)
        if s["stage"] not in ("editing", "simulating"):
            raise InputError("Confirm the Photo A alignment first")
        arch = library.load_arch_npz(storage() / f"session_{s['id']}" / "working_model.npz")
        core.apply_tooth_offsets(arch, body.get("tooth_offsets") or {})  # validates (single mesh / unknown tooth)
        v = {"n": len(s["versions"]) + 1, "note": str(body.get("note", ""))[:200],
             "tooth_offsets": body.get("tooth_offsets") or {}, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        s["versions"].append(v)
        persist(s)
        return view(s)

    @r.post("/sessions/{i}/versions/{n}/restore")
    def restore(i: str, n: int):
        s = sess(i)
        v = next((x for x in s["versions"] if x["n"] == n), None)
        if v is None:
            raise KeyError(f"version {n}")
        s["versions"].append({**v, "n": len(s["versions"]) + 1, "note": f"restored from v{n}"})
        persist(s)
        return view(s)

    @r.put("/sessions/{i}/photo_b")
    def photo_b(i: str, body: dict):
        """Photo B: cropped mouth region + its own, independent pose (separate from the Photo A alignment)."""
        s = sess(i)
        if s["stage"] not in ("editing", "simulating"):
            raise InputError("Confirm the Photo A alignment and save the edited model first")
        if not s["photo_b"]:
            raise InputError("Upload Photo B first")
        pb = s["photo_b"]
        if body.get("crop"):
            x, y, w, h = [float(v) for v in body["crop"]]
            if not (0 <= x and 0 <= y and w > 10 and h > 10 and x + w <= pb["width"] + 1 and y + h <= pb["height"] + 1):
                raise InputError("crop outside Photo B")
        if body.get("pose"):
            core.Pose.from_json(body["pose"])
        s["photo_b_placement"] = {k: body.get(k) for k in ("crop", "pose", "fov_deg", "opacity", "layer", "mask_polygons", "version")}
        s["stage"] = "simulating"
        persist(s)
        return view(s)

    return r
