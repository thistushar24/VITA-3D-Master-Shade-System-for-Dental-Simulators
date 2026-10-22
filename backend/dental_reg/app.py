"""FastAPI service. All state is in-process + files under STORAGE_DIR (research prototype, no DB).

Responses: {"ok": true, ...} on success; errors are {"ok": false, "error": "<message>"} with 4xx/5xx.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import trimesh
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import core, library
from . import stateless as stateless_mod
from . import v1 as v1mod
from .core import InputError, Pose, RegConfig

ROOT = Path(__file__).resolve().parents[2]
ON_VERCEL = bool(os.environ.get("VERCEL"))  # deployment filesystem is read-only except /tmp, which is per-instance and not durable
STORAGE = Path(os.environ.get("DENTAL_STORAGE", "/tmp/dental-storage" if ON_VERCEL else ROOT / "backend" / "storage"))
MODEL_ROOTS = [Path(p) for p in os.environ.get("DENTAL_MODEL_ROOTS", str(ROOT / "data" / "samples")).split(os.pathsep)]
LIB_DIR = lambda: Path(os.environ.get("DENTAL_LIBRARY_DIR", ROOT / "data" / "demo_library"))
MAX_IMAGE_BYTES = int(float(os.environ.get("DENTAL_MAX_IMAGE_MB", 4 if ON_VERCEL else 20)) * 2**20)  # Vercel caps request bodies at 4.5 MB
MAX_MESH_BYTES = int(os.environ.get("DENTAL_MAX_MESH_MB", 100)) * 2**20
TTL_SECONDS = 24 * 3600
SCHEMA_VERSION = "dental-alignment/1"

app = FastAPI(title="Dental 2D-3D registration", version="0.1")
# CORS: explicit origins via DENTAL_CORS (comma list); optional DENTAL_CORS_REGEX for e.g. Vercel preview URLs. No credentials.
app.add_middleware(CORSMiddleware, allow_origins=[o for o in os.environ.get("DENTAL_CORS", "http://localhost:5173,http://localhost:5174").split(",") if o],
                   allow_origin_regex=os.environ.get("DENTAL_CORS_REGEX") or None, allow_methods=["*"], allow_headers=["*"])

REGS: dict[str, dict] = {}
IMAGES: dict[str, dict] = {}


@app.exception_handler(InputError)
async def _input_error(_: Request, e: InputError):
    return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.exception_handler(KeyError)
async def _missing(_: Request, e: KeyError):
    return JSONResponse({"ok": False, "error": f"not found: {e.args[0]}"}, status_code=404)


@app.on_event("startup")
def _cleanup_old():
    """Delete uploads/exports older than TTL (temporary-file cleanup)."""
    STORAGE.mkdir(parents=True, exist_ok=True)
    for p in STORAGE.iterdir():
        if time.time() - p.stat().st_mtime > TTL_SECONDS:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


def _new_dir(kind: str) -> tuple[str, Path]:
    i = uuid.uuid4().hex[:12]
    d = STORAGE / f"{kind}_{i}"
    d.mkdir(parents=True, exist_ok=True)
    return i, d


async def _read_limited(f: UploadFile, limit: int) -> bytes:
    data = await f.read(limit + 1)
    if len(data) > limit:
        raise InputError(f"file too large (limit {limit // 2**20} MB)")
    if not data:
        raise InputError("empty file")
    return data


# ----------------------------------------------------------------------------- models
def list_models() -> list[dict]:
    out = []
    for root in MODEL_ROOTS:
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if any(d.glob("tooth_*.obj")):
                jaw = "upper" if any(d.glob("tooth_1*.obj")) or any(d.glob("tooth_2*.obj")) else "lower"
                out.append({"id": f"case:{d.parent.name}/{d.name}", "name": d.name, "jaw": jaw,
                            "format": "obj (per-tooth, FDI-labelled)", "teeth": len(list(d.glob("tooth_*.obj")))})
    for d in sorted(STORAGE.glob("model_*")):
        f = next((p for p in d.iterdir() if p.suffix.lower() in core.MESH_EXTS), None)
        if f:
            out.append({"id": f"upload:{d.name[6:]}", "name": f.name, "jaw": "unknown",
                        "format": f.suffix[1:] + " (single mesh)", "teeth": 0})
    return out


@lru_cache(maxsize=8)
def _arch(model_id: str) -> core.Arch:
    kind, _, ref = model_id.partition(":")
    if kind == "case":
        for root in MODEL_ROOTS:
            d = (root.parent / ref).resolve()
            if d.is_dir() and root.resolve() in d.parents:  # no path traversal out of the model root
                return core.load_arch_dir(d)
        raise KeyError(model_id)
    if kind == "lib":
        ent = next((e for e in library.load_catalogue(LIB_DIR()) if e["id"] == ref), None)  # id must be in catalogue
        if ent is None:
            raise KeyError(model_id)
        return library.load_arch_npz(LIB_DIR() / ent["file"])
    if kind == "upload" and ref.isalnum():
        d = STORAGE / f"model_{ref}"
        f = next((p for p in d.iterdir() if p.suffix.lower() in core.MESH_EXTS), None) if d.is_dir() else None
        if f:
            return core.load_arch_file(f)
    raise KeyError(model_id)


@app.get("/api/health")
def health():
    return {"ok": True, "version": app.version, "runtime": "vercel" if ON_VERCEL else "standalone"}


@app.get("/api/models")
def models():
    return {"ok": True, "models": list_models()}


@app.post("/api/models")
async def upload_model(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in core.MESH_EXTS:
        raise InputError(f"unsupported mesh format {ext!r}; use OBJ, PLY or STL")
    data = await _read_limited(file, MAX_MESH_BYTES)
    i, d = _new_dir("model")
    path = d / f"model{ext}"
    path.write_bytes(data)
    try:
        core.load_mesh(path)  # validate now
    except InputError:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return {"ok": True, "id": f"upload:{i}"}


@app.get("/api/models/{kind}/{ref:path}/mesh")
def model_mesh(kind: str, ref: str, target_faces: int = 12000):
    """Decimated mesh for display. Vertices are in original model coordinates (mm)."""
    a = core.decimate(_arch(f"{kind}:{ref}"), max(500, min(target_faces, 200000)))
    return {"ok": True, "units": "mm", "jaw": a.jaw, "name": a.name, "center": a.center.tolist(),
            "vertices": np.round(a.vertices, 3).ravel().tolist(), "faces": a.faces.ravel().tolist(),
            "face_tooth": a.face_tooth.tolist(),
            "teeth": {str(k): v.tolist() for k, v in a.teeth.items()},
            "tooth_editing": a.has_teeth,
            "tooth_editing_note": None if a.has_teeth else "Single mesh: individual teeth are not separable, tooth editing disabled."}


@app.get("/api/samples/{name}")
def sample(name: str):
    """Synthetic demo files shipped in data/samples (photo, mask, annotations, ground truth)."""
    if name not in {"synthetic_photo.png", "synthetic_annotations.json", "synthetic_ground_truth.json", "demo_photo_a.png", "demo_photo_b.png", "demo_annotations.json"}:
        raise KeyError(name)
    return FileResponse(ROOT / "data" / "samples" / name)


# ----------------------------------------------------------------------------- images
@app.post("/api/images")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in core.IMAGE_EXTS:
        raise InputError(f"unsupported image format {ext!r}; use PNG or JPEG")
    data = await _read_limited(file, MAX_IMAGE_BYTES)
    i, d = _new_dir("image")
    path = d / f"photo{ext}"
    path.write_bytes(data)
    try:
        img = core.load_image(path)
    except InputError:
        shutil.rmtree(d, ignore_errors=True)
        raise
    IMAGES[i] = {"path": path, "name": file.filename}
    (d / "name.txt").write_text(file.filename or "")
    return {"ok": True, "id": i, "width": img.shape[1], "height": img.shape[0], "name": file.filename}


def _image(i):
    if i not in IMAGES and isinstance(i, str) and i.isalnum():  # re-adopt an upload that survived a restart
        d = STORAGE / f"image_{i}"
        f = next((p for p in d.glob("photo.*")), None) if d.is_dir() else None
        if f:
            nm = d / "name.txt"
            IMAGES[i] = {"path": f, "name": nm.read_text() if nm.is_file() else f.name}
    if i not in IMAGES:
        raise InputError(f"unknown image id {i!r}")
    return IMAGES[i]


@app.get("/api/images/{i}/file")
def image_file(i: str):
    return FileResponse(_image(i)["path"])


# ----------------------------------------------------------------------------- registrations
def _annotations(body: dict, w: int, h: int) -> dict:
    markers = body.get("markers") or []
    for m in markers:
        core.validate_points([m["image_xy"]], w, h, f"marker {m.get('id')}")
    landmarks = body.get("landmarks") or []
    for lm in landmarks:
        core.validate_points([lm["image_xy"]], w, h, f"landmark tooth {lm.get('fdi')}")
    polys = body.get("mask_polygons") or []
    for p in polys:
        core.validate_points(p, w, h, "mask polygon")
    return {"markers": markers, "landmarks": landmarks, "mask_polygons": polys}


def _persist_reg(r: dict):
    """Write registration state to STORAGE so it survives a restart (best effort)."""
    try:
        doc = {k: v for k, v in r.items() if k not in ("pose", "K")} | {"pose": r["pose"].to_json(), "K": np.asarray(r["K"]).tolist()}
        (STORAGE / f"reg_{r['id']}.json").write_text(json.dumps(doc))
    except OSError:
        pass


def _reg(i) -> dict:
    if i not in REGS and isinstance(i, str) and i.isalnum():
        f = STORAGE / f"reg_{i}.json"
        if f.is_file():
            d = json.loads(f.read_text())
            d["pose"], d["K"] = Pose.from_json(d["pose"]), np.array(d["K"])
            REGS[i] = d
    if i not in REGS:
        raise KeyError(i)
    return REGS[i]


def _view(r: dict) -> dict:
    _persist_reg(r)
    return {"ok": True, "id": r["id"], "status": r["status"], "image": r["image"], "model": r["model"],
            "camera": r["camera"], "annotations": r["annotations"], "pose": r["pose"].to_json(),
            "tooth_offsets": r["tooth_offsets"], "report": r["report"], "mode": r["mode"]}


def _camera_json(K, w, h, approx):
    return {"K": np.asarray(K).tolist(), "width": w, "height": h, "calibrated": not approx,
            "projection": "perspective", **({"fov_deg": r_fov(K, w)} if approx else {})}


def r_fov(K, w):
    return float(np.degrees(2 * np.arctan((w / 2) / K[0][0])))


@app.post("/api/registrations")
def create_registration(body: dict):
    """body: {image_id, model_id, camera?: {fov_deg | K}, annotations?: {markers, landmarks, mask_polygons}}"""
    im = _image(body.get("image_id"))
    arch = _arch(body.get("model_id", ""))
    img = core.load_image(im["path"])
    h, w = img.shape[:2]
    cam = body.get("camera") or {}
    K = core.intrinsics(w, h, cam.get("fov_deg", 40.0), cam.get("K"))
    i = uuid.uuid4().hex[:12]
    REGS[i] = {"id": i, "status": "created", "mode": "initial", "image_id": body["image_id"],
               "image": {"id": body["image_id"], "name": im["name"], "width": w, "height": h},
               "model_id": body["model_id"], "model": {"id": body["model_id"], "name": arch.name, "jaw": arch.jaw,
                                                        "units": "mm", "teeth": sorted(arch.teeth)},
               "camera": _camera_json(K, w, h, cam.get("K") is None), "K": K,
               "annotations": _annotations(body.get("annotations") or {}, w, h),
               "pose": core.initial_pose(arch, w, h, K), "tooth_offsets": {}, "report": None}
    return _view(REGS[i])


@app.get("/api/registrations/{i}")
def get_registration(i: str):
    return _view(_reg(i))


@app.put("/api/registrations/{i}/annotations")
def set_annotations(i: str, body: dict):
    r = _reg(i)
    r["annotations"] = _annotations(body, r["image"]["width"], r["image"]["height"])
    return _view(r)


@app.post("/api/registrations/{i}/auto")
def auto_align(i: str, body: dict | None = None):
    """Run automatic registration from landmarks and/or mask polygons. Options in body['config'] (RegConfig)."""
    r = _reg(i)
    cfg = RegConfig.from_json((body or {}).get("config"))
    w, h = r["image"]["width"], r["image"]["height"]
    a = r["annotations"]
    arch = _arch(r["model_id"])
    mask = core.mask_from_polygons(a["mask_polygons"], w, h) if a["mask_polygons"] else None
    lm = [(int(x["fdi"]), tuple(x["image_xy"])) for x in a["landmarks"]]
    V = core.apply_tooth_offsets(arch, r["tooth_offsets"])
    pose, rep = core.register(arch, (h, w), r["K"], landmarks=lm, mask=mask, cfg=cfg, vertices=V)
    r.update(pose=pose, report=rep, mode="auto", status=rep["status"])
    return _view(r)


@app.post("/api/registrations/{i}/transform")
def set_transform(i: str, body: dict):
    """Manual alignment: body {pose?, tooth_offsets?}. Metrics recomputed from the annotations when available."""
    r = _reg(i)
    if "pose" in body:
        r["pose"] = Pose.from_json(body["pose"])
    if "tooth_offsets" in body:
        arch = _arch(r["model_id"])
        core.apply_tooth_offsets(arch, body["tooth_offsets"])  # validates
        r["tooth_offsets"] = body["tooth_offsets"]
    r["mode"], r["status"] = "manual", "manual"
    r["report"] = {"method": "manual", "status": "manual", "metrics": _metrics(r), "notes":
                   ["Manually adjusted alignment; metrics are measured against your annotations only."]}
    return _view(r)


def _metrics(r) -> dict:
    a, arch = r["annotations"], _arch(r["model_id"])
    w, h = r["image"]["width"], r["image"]["height"]
    V = core.apply_tooth_offsets(arch, r["tooth_offsets"])
    out = {}
    lm = [x for x in a["landmarks"] if int(x["fdi"]) in arch.teeth]
    if lm:
        uv, _, ok = core.project(np.array([arch.teeth[int(x["fdi"])] for x in lm]), r["pose"], r["K"])
        d = np.linalg.norm(uv - np.array([x["image_xy"] for x in lm]), axis=1)
        out["landmark_rmse_px"] = float(np.sqrt(np.mean(d**2))) if ok.all() else None
    if a["mask_polygons"]:
        mask = core.mask_from_polygons(a["mask_polygons"], w, h)
        out["silhouette_iou"] = core.iou(core.silhouette(V, arch.faces, r["pose"], r["K"], w, h, arch.face_tooth), mask)
    return out


@app.post("/api/registrations/{i}/export")
def export(i: str):
    """Write alignment JSON + aligned OBJ/PLY under STORAGE/export_<id>; originals are never modified."""
    r = _reg(i)
    arch = _arch(r["model_id"])
    V = core.apply_tooth_offsets(arch, r["tooth_offsets"])
    M = r["pose"].matrix()
    Vc = V @ M[:3, :3].T + M[:3, 3]  # camera-frame coordinates (mm)
    d = STORAGE / f"export_{r['id']}"
    d.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(Vc, arch.faces, process=False)
    mesh.export(d / "aligned_model.obj")
    mesh.export(d / "aligned_model.ply")
    doc = alignment_document(r)
    (d / "alignment.json").write_text(json.dumps(doc, indent=2))
    return {"ok": True, "alignment": doc, "files": ["alignment.json", "aligned_model.obj", "aligned_model.ply"]}


def alignment_document(r) -> dict:
    rep = r["report"] or {}
    return {"schema": SCHEMA_VERSION, "status": r["status"], "mode": r["mode"],
            "disclaimer": "Fitted/estimated template alignment from a single photograph; not a reconstruction of the "
                          "patient's true anatomy and not for clinical use.",
            "image": r["image"], "model": r["model"], "camera": r["camera"], "pose": r["pose"].to_json(),
            "tooth_offsets": r["tooth_offsets"], "annotations": r["annotations"],
            "metrics": rep.get("metrics"), "method": rep.get("method"), "notes": rep.get("notes", []),
            "config": rep.get("config"),
            "conventions": "X_cam = scale * R(rvec) @ (X_model - center) + t ; OpenCV camera (+x right, +y down, +z forward); mm",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


@app.get("/api/registrations/{i}/export/{name}")
def export_file(i: str, name: str):
    if name not in {"alignment.json", "aligned_model.obj", "aligned_model.ply"}:
        raise InputError("unknown export file")
    p = STORAGE / f"export_{_reg(i)['id']}" / name
    if not p.is_file():
        raise InputError("export first (POST /api/registrations/{id}/export)")
    return FileResponse(p, filename=name)


@app.post("/api/registrations/load")
def load_alignment(body: dict):
    """Re-create a registration from a saved alignment JSON (needs the image to be uploaded again: body.image_id)."""
    if body.get("alignment", {}).get("schema") != SCHEMA_VERSION:
        raise InputError(f"not a {SCHEMA_VERSION} document")
    doc = body["alignment"]
    r = create_registration({"image_id": body.get("image_id"), "model_id": doc["model"]["id"],
                             "camera": {"K": doc["camera"]["K"]}, "annotations": doc.get("annotations")})
    reg = REGS[r["id"]]
    reg["pose"] = Pose.from_json(doc["pose"])
    reg["tooth_offsets"] = doc.get("tooth_offsets") or {}
    reg["mode"], reg["status"] = doc.get("mode", "manual"), "loaded"
    return _view(reg)


app.include_router(v1mod.make_router(
    images=IMAGES, storage=lambda: STORAGE, lib_dir=LIB_DIR,
    create_reg=lambda body: create_registration(body), auto_align=lambda i, body: auto_align(i, body),
    set_transform=lambda i, body: set_transform(i, body), get_reg=lambda i: get_registration(i)))

app.include_router(stateless_mod.make_router(
    load_arch=lambda mid: _arch(mid), lib_dir=LIB_DIR, read_limited=_read_limited, max_image_bytes=MAX_IMAGE_BYTES))

# Single-service deployment: serve the built v1 demo UI when present (see render.yaml). API routes win over static.
# Otherwise (API-only deployment, e.g. a separate Vercel project) GET / returns a small service description.
_DIST = Path(os.environ.get("DENTAL_UI_DIR", ROOT / "versions" / "v1" / "dist"))
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="v1-ui")
else:
    @app.get("/", include_in_schema=False)
    def root_info():
        return {"ok": True, "service": "dental-reg api", "health": "/api/health", "catalogue": "/api/v1/catalogue", "docs": "/docs"}
