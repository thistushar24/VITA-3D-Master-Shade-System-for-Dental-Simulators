"""Stateless compute API for serverless hosting (Vercel): no sessions, no server-side files that must outlive a request.

The browser keeps the session; each endpoint is a pure function of its request:
  POST /api/v1/stateless/analyse    multipart photo A  -> tooth region + estimated canines (pixels needed only here)
  POST /api/v1/stateless/match      JSON annotations   -> ranked catalogue, selected model, auto-placed registration
  POST /api/v1/stateless/place      JSON + model_id    -> registration for a user-chosen model
  POST /api/v1/stateless/transform  JSON registration  -> same registration with a manual pose/tooth offsets + metrics
Registration objects have the same shape as the stateful /api/registrations responses (minus server ids/state).
"""
from __future__ import annotations

import uuid

import numpy as np
from fastapi import APIRouter, File, UploadFile

from . import core, library, match, vision
from .core import InputError, Pose, RegConfig

MAX_DIM = 6000


def make_router(*, load_arch, lib_dir, read_limited, max_image_bytes) -> APIRouter:
    r = APIRouter(prefix="/api/v1/stateless")

    def dims(body):
        try:
            w, h = int(body["width"]), int(body["height"])
        except (KeyError, TypeError, ValueError) as e:
            raise InputError("width and height (pixels of Photo A) are required") from e
        if not (16 <= w <= MAX_DIM and 16 <= h <= MAX_DIM):
            raise InputError(f"image size must be between 16 and {MAX_DIM}px")
        return w, h

    def annotations(body, w, h):
        canines = body.get("canines")
        if canines is not None:
            canines = core.validate_points(canines, w, h, "canines").tolist()
            if len(canines) != 2:
                raise InputError("exactly two canine points are required (patient's right = image left, then patient's left)")
        polys = body.get("mask_polygons") or []
        for p in polys:
            core.validate_points(p, w, h, "mask polygon")
        return canines, polys

    def build(model_id, w, h, canines, polys, fov, auto=True):
        arch = load_arch(model_id)
        K = core.intrinsics(w, h, fov)
        pose, report = core.initial_pose(arch, w, h, K), None
        lm = [(13, tuple(canines[0])), (23, tuple(canines[1]))] if canines and 13 in arch.teeth and 23 in arch.teeth else []
        if auto:
            mask = core.mask_from_polygons(polys, w, h) if polys else None
            try:
                pose, report = core.register(arch, (h, w), K, landmarks=lm, mask=mask, cfg=RegConfig())
            except InputError as e:  # nothing to fit with: keep the initial pose and say so
                report = {"method": "initial", "status": "initial", "metrics": {}, "notes": [f"automatic placement skipped: {e}"]}
        return {"id": uuid.uuid4().hex[:12], "status": (report or {}).get("status", "created"), "mode": "auto" if auto else "initial",
                "image": {"width": w, "height": h},
                "model": {"id": model_id, "name": arch.name, "jaw": arch.jaw, "units": "mm", "teeth": sorted(arch.teeth)},
                "camera": {"K": K.tolist(), "width": w, "height": h, "calibrated": False, "projection": "perspective", "fov_deg": fov},
                "annotations": {"markers": [], "landmarks": [{"fdi": 13, "image_xy": canines[0]}, {"fdi": 23, "image_xy": canines[1]}] if canines else [],
                                "mask_polygons": polys},
                "pose": pose.to_json(), "tooth_offsets": {}, "report": report}

    def fov_of(body):
        try:
            return float(body.get("fov_deg", 40.0))
        except (TypeError, ValueError) as e:
            raise InputError("fov_deg must be a number") from e

    @r.post("/analyse")
    async def analyse(file: UploadFile = File(...)):
        data = await read_limited(file, max_image_bytes)
        import cv2
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise InputError("Photograph could not be decoded; use PNG or JPEG")
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            raise InputError(f"image too large ({w}x{h}); downscale below {MAX_DIM}px")
        out = {"ok": True, "width": w, "height": h, "name": file.filename}
        try:
            seg = vision.segment_teeth(img)
            out["analysis"] = {"mask_polygons": vision.contours_from_mask(seg["mask_upper"]), "canines": seg["canines"],
                               "notes": seg["notes"], "confidence": seg["confidence"], "source": "automatic"}
        except InputError as e:
            out["analysis"] = {"mask_polygons": [], "canines": None, "notes": [str(e)], "confidence": "none"}
        return out

    @r.post("/match")
    def run_match(body: dict):
        w, h = dims(body)
        canines, polys = annotations(body, w, h)
        if not canines:
            raise InputError("Canine reference points are needed: drag the two points onto the canines")
        if not polys:
            raise InputError("No tooth region: run analysis or draw an outline")
        fov = fov_of(body)
        cat = library.load_catalogue(lib_dir())
        res = match.rank_models(core.mask_from_polygons(polys, w, h), canines, cat, lib_dir(), (h, w, 3), fov_deg=fov,
                                top_k=int(body.get("top_k", 6)))
        return {"ok": True, "match": res, "model_id": f"lib:{res['selected']}", "fov_deg": fov,
                "registration": build(f"lib:{res['selected']}", w, h, canines, polys, fov)}

    @r.post("/place")
    def place(body: dict):
        w, h = dims(body)
        canines, polys = annotations(body, w, h)
        ids = {e["id"] for e in library.load_catalogue(lib_dir())}
        mid = body.get("model_id", "")
        if mid.removeprefix("lib:") not in ids:
            raise InputError(f"model {mid!r} is not in the catalogue")
        return {"ok": True, "model_id": f"lib:{mid.removeprefix('lib:')}",
                "registration": build(f"lib:{mid.removeprefix('lib:')}", w, h, canines, polys, fov_of(body))}

    @r.post("/transform")
    def transform(body: dict):
        reg = body.get("registration")
        if not isinstance(reg, dict):
            raise InputError("registration object required")
        try:
            w, h = int(reg["image"]["width"]), int(reg["image"]["height"])
            K = np.array(reg["camera"]["K"], float)
            mid = reg["model"]["id"]
        except (KeyError, TypeError, ValueError) as e:
            raise InputError(f"malformed registration: {e}") from e
        arch = load_arch(mid)
        pose = Pose.from_json(body["pose"]) if "pose" in body else Pose.from_json(reg["pose"])
        offsets = body.get("tooth_offsets", reg.get("tooth_offsets")) or {}
        V = core.apply_tooth_offsets(arch, offsets)  # validates tooth ids / single-mesh
        ann = reg.get("annotations") or {}
        metrics = {}
        lm = [x for x in ann.get("landmarks", []) if int(x["fdi"]) in arch.teeth]
        if lm:
            uv, _, ok = core.project(np.array([arch.teeth[int(x["fdi"])] for x in lm]), pose, K)
            d = np.linalg.norm(uv - np.array([x["image_xy"] for x in lm], float), axis=1)
            metrics["landmark_rmse_px"] = float(np.sqrt(np.mean(d ** 2))) if ok.all() else None
        if ann.get("mask_polygons"):
            mask = core.mask_from_polygons(ann["mask_polygons"], w, h)
            metrics["silhouette_iou"] = core.iou(core.silhouette(V, arch.faces, pose, K, w, h, arch.face_tooth), mask)
        out = dict(reg) | {"pose": pose.to_json(), "tooth_offsets": offsets, "mode": "manual", "status": "manual",
                           "report": {"method": "manual", "status": "manual", "metrics": metrics,
                                      "notes": ["Manually adjusted alignment; metrics are measured against your annotations only."]}}
        return {"ok": True, "registration": out}

    return r
