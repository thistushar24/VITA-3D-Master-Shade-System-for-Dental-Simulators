"""Loading/validation, camera projection, and 2D-3D registration for dental arches.

Conventions (see docs/DATA_CONTRACT.md):
  * Model space: millimetres, Teeth3DS scanner frame, untouched on disk. We subtract the
    arch centroid `center` (kept in the pose) so rotation happens about the arch.
  * Pose: X_cam = scale * R @ (X_model - center) + t.  Camera looks along +Z, +X right, +Y down
    (OpenCV convention). Image pixels: origin top-left, x right, y down.
  * Frontal reference rotation R_FRONT: camera sits in front of the incisors (-Y model), model
    right (+X, FDI 1x/4x quadrants) appears on image left, FDI-patient-left on image right.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MESH_EXTS = {".obj", ".ply", ".stl"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
# model +x -> image left, model +y (toward molars) -> away from camera, model -z -> image up
R_FRONT = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
FDI_RE = re.compile(r"tooth_(\d\d)\.", re.I)


class InputError(ValueError):
    """Raised for missing/invalid/incompatible inputs; message is user-facing."""


# ----------------------------------------------------------------------------- loading
def load_image(path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"Photograph not found: {p}")
    if p.suffix.lower() not in IMAGE_EXTS:
        raise InputError(f"Unsupported image format {p.suffix!r}; use PNG or JPEG")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise InputError(f"Photograph could not be decoded: {p.name}")
    return img


def validate_points(pts, width: int, height: int, name="points") -> np.ndarray:
    a = np.asarray(pts, dtype=float).reshape(-1, 2) if len(pts) else np.zeros((0, 2))
    if not np.isfinite(a).all():
        raise InputError(f"{name}: non-finite coordinates")
    bad = (a[:, 0] < 0) | (a[:, 0] > width - 1) | (a[:, 1] < 0) | (a[:, 1] > height - 1)
    if bad.any():
        raise InputError(f"{name}: {int(bad.sum())} point(s) outside image bounds {width}x{height}")
    return a


def validate_mesh(m: trimesh.Trimesh, name="mesh") -> trimesh.Trimesh:
    if len(m.vertices) == 0 or len(m.faces) == 0:
        raise InputError(f"{name}: empty mesh (no vertices or faces)")
    if not np.isfinite(m.vertices).all():
        raise InputError(f"{name}: mesh contains NaN/inf vertices")
    if m.faces.max() >= len(m.vertices) or m.faces.min() < 0:
        raise InputError(f"{name}: face indices out of range")
    ext = float(np.ptp(m.vertices, axis=0).max())
    if ext < 1e-6:
        raise InputError(f"{name}: degenerate mesh (zero extent)")
    return m


def load_mesh(path) -> trimesh.Trimesh:
    p = Path(path)
    if p.suffix.lower() not in MESH_EXTS:
        raise InputError(f"Unsupported mesh format {p.suffix!r}; use OBJ, PLY or STL")
    if not p.is_file():
        raise InputError(f"Mesh not found: {p}")
    with open(p, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            raise InputError(f"{p.name} is a Git LFS pointer, not mesh data. Run `git lfs pull`.")
    try:
        m = trimesh.load(p, force="mesh", process=False)
    except Exception as e:  # trimesh raises many types for malformed files
        raise InputError(f"Mesh could not be parsed ({p.name}): {e}") from e
    if not isinstance(m, trimesh.Trimesh):
        raise InputError(f"{p.name}: not a triangle mesh")
    return validate_mesh(m, p.name)


@dataclass
class Arch:
    """A dental arch: merged triangle mesh + per-face FDI label (0 = unlabelled single mesh)."""
    name: str
    jaw: str
    vertices: np.ndarray
    faces: np.ndarray
    face_tooth: np.ndarray
    teeth: dict = field(default_factory=dict)  # fdi -> centroid (mm)

    @property
    def center(self):
        return self.vertices.mean(0)

    @property
    def has_teeth(self):
        return len(self.teeth) > 1

    def normals(self):
        return trimesh.Trimesh(self.vertices, self.faces, process=False).vertex_normals


def build_arch(parts: dict[int, trimesh.Trimesh], name: str, jaw: str) -> Arch:
    vs, fs, ft, off, teeth = [], [], [], 0, {}
    for fdi, m in sorted(parts.items()):
        vs.append(np.asarray(m.vertices, float))
        fs.append(np.asarray(m.faces) + off)
        ft.append(np.full(len(m.faces), fdi))
        off += len(m.vertices)
        if fdi:
            teeth[fdi] = np.asarray(m.vertices, float).mean(0)
    return Arch(name, jaw, np.vstack(vs), np.vstack(fs), np.concatenate(ft), teeth)


def load_arch_dir(case_dir, jaw=None) -> Arch:
    """Load Person 2's per-tooth OBJ folder (tooth_<FDI>.obj + optional metadata.json)."""
    d = Path(case_dir)
    if not d.is_dir():
        raise InputError(f"Model case folder not found: {d}")
    parts = {}
    for f in sorted(d.glob("tooth_*.obj")):
        mt = FDI_RE.match(f.name)
        if mt:
            parts[int(mt.group(1))] = load_mesh(f)
    if not parts:
        raise InputError(f"No tooth_<FDI>.obj files in {d}")
    if jaw is None:
        meta = d / "metadata.json"
        jaw = json.loads(meta.read_text()).get("jaw") if meta.is_file() else None
        jaw = jaw or ("upper" if min(parts) < 30 else "lower")
    return build_arch(parts, d.name, jaw)


def load_arch_file(path, jaw="unknown") -> Arch:
    return build_arch({0: load_mesh(path)}, Path(path).stem, jaw)


def decimate(arch: Arch, target_faces: int) -> Arch:
    """Per-tooth quadric decimation (keeps labels); returns arch unchanged if already small."""
    if len(arch.faces) <= target_faces:
        return arch
    import fast_simplification as fs
    parts = {}
    labels = np.unique(arch.face_tooth)
    ratio = 1 - target_faces / len(arch.faces)
    for fdi in labels:
        sel = arch.faces[arch.face_tooth == fdi]
        used, inv = np.unique(sel, return_inverse=True)
        v, f = fs.simplify(arch.vertices[used].astype(np.float32), inv.reshape(-1, 3).astype(np.int32),
                           target_reduction=min(ratio, 0.97))
        parts[int(fdi)] = trimesh.Trimesh(v.astype(float), f, process=False)
    return build_arch(parts, arch.name, arch.jaw)


# ----------------------------------------------------------------------------- camera
def intrinsics(width: int, height: int, fov_deg: float = 40.0, K=None) -> np.ndarray:
    """Calibrated K if given, otherwise an *approximate* pinhole (horizontal FOV, centred principal point)."""
    if K is not None:
        K = np.asarray(K, float)
        if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise InputError("camera.K must be a valid 3x3 intrinsic matrix")
        return K
    if not 5 <= fov_deg <= 120:
        raise InputError("fov_deg must be between 5 and 120")
    f = (width / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


@dataclass
class Pose:
    rvec: np.ndarray  # rotation vector, model->camera
    t: np.ndarray  # mm, camera frame
    scale: float
    center: np.ndarray  # model-space pivot subtracted before rotating

    def R(self):
        return Rotation.from_rotvec(self.rvec).as_matrix()

    def matrix(self) -> np.ndarray:
        """4x4 model->camera (includes the pivot shift and scale)."""
        M = np.eye(4)
        M[:3, :3] = self.scale * self.R()
        M[:3, 3] = self.t - self.scale * self.R() @ self.center
        return M

    def to_json(self):
        return {"rvec": self.rvec.tolist(), "t": self.t.tolist(), "scale": float(self.scale),
                "center": self.center.tolist(), "model_to_camera": self.matrix().tolist()}

    @staticmethod
    def from_json(d):
        try:
            rvec = Rotation.from_matrix(np.asarray(d["R"], float)).as_rotvec() if "R" in d else d["rvec"]
            p = Pose(np.asarray(rvec, float), np.asarray(d["t"], float), float(d["scale"]),
                     np.asarray(d["center"], float))
        except (KeyError, TypeError, ValueError) as e:
            raise InputError(f"invalid pose: {e}") from e
        if p.rvec.shape != (3,) or p.t.shape != (3,) or p.center.shape != (3,):
            raise InputError("invalid pose: rvec/t/center must have 3 components")
        if not np.isfinite(np.r_[p.rvec, p.t, p.scale, p.center]).all() or not 0.2 <= p.scale <= 5:
            raise InputError("invalid pose: non-finite values or scale outside [0.2, 5]")
        return p


def project(points, pose: Pose, K, near: float = 1.0, orthographic: bool = False):
    """Project Nx3 model points. Returns (uv Nx2, depth N, valid N). Behind-camera points invalid."""
    X = pose.scale * (np.asarray(points, float) - pose.center) @ pose.R().T + pose.t
    z = X[:, 2]
    valid = z > near
    if orthographic:
        s = K[0, 0] / max(pose.t[2], near)  # px per mm at the arch depth
        uv = np.c_[K[0, 2] + s * X[:, 0], K[1, 2] + s * X[:, 1]]
    else:
        zs = np.where(valid, z, 1.0)
        uv = np.c_[K[0, 0] * X[:, 0] / zs + K[0, 2], K[1, 1] * X[:, 1] / zs + K[1, 2]]
    return uv, z, valid


def apply_tooth_offsets(arch: Arch, offsets: dict | None) -> np.ndarray:
    """Return vertices with per-tooth edits applied: rotate (deg, XYZ euler) about the tooth centroid, then translate (mm).
    Disabled (error) for single-mesh models."""
    V = arch.vertices.copy()
    if not offsets:
        return V
    if not arch.has_teeth:
        raise InputError("Individual tooth editing needs a model with separate tooth meshes")
    face_v = {}
    for fdi in offsets:
        fdi_i = int(fdi)
        if fdi_i not in arch.teeth:
            raise InputError(f"Unknown tooth {fdi}")
        idx = np.unique(arch.faces[arch.face_tooth == fdi_i])
        o = offsets[fdi]
        Rm = Rotation.from_euler("xyz", o.get("r", [0, 0, 0]), degrees=True).as_matrix()
        c = arch.teeth[fdi_i]
        V[idx] = (V[idx] - c) @ Rm.T + c + np.asarray(o.get("t", [0, 0, 0]), float)
        face_v[fdi_i] = idx
    return V


def initial_pose(arch: Arch, width: int, height: int, K) -> Pose:
    """Frontal reference pose, arch centred in the image at a depth giving ~60% image width."""
    c = arch.center
    ext = np.ptp(arch.vertices @ R_FRONT.T, axis=0)  # extents in camera axes
    tz = K[0, 0] * ext[0] / (0.6 * width)
    return Pose(Rotation.from_matrix(R_FRONT).as_rotvec(), np.array([0.0, 0.0, tz]), 1.0, c)


# ----------------------------------------------------------------------------- annotations
def mask_from_polygons(polys, width, height) -> np.ndarray:
    m = np.zeros((height, width), np.uint8)
    for p in polys:
        a = validate_points(p, width, height, "mask polygon")
        if len(a) < 3:
            raise InputError("mask polygon needs >= 3 points")
        cv2.fillPoly(m, [np.round(a).astype(np.int32)], 255)
    return m


def load_mask(path, width, height) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise InputError(f"Mask could not be decoded: {path}")
    if m.shape != (height, width):
        raise InputError(f"Mask size {m.shape[::-1]} != image size {(width, height)}")
    return ((m > 127) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------- registration
@dataclass
class RegConfig:
    """Objective weights/safeguards. Residual blocks (all in pixels): landmark reprojection,
    model->mask boundary distance, mask->model coverage distance; robust loss on all of them."""
    w_landmark: float = 2.0
    w_model_to_mask: float = 1.0
    w_mask_to_model: float = 1.0
    w_scale_prior: float = 20.0  # px per unit |log scale|; keeps scale near 1 (mm model)
    loss: str = "soft_l1"
    f_scale: float = 8.0  # px, robust-loss knee
    max_rot_deg: float = 35.0  # allowed rotation from initial pose
    scale_bounds: tuple = (0.8, 1.25)
    n_model_pts: int = 800
    n_mask_pts: int = 500
    max_nfev: int = 200
    min_iou_gain: float = 0.0  # refined result is rejected if it does not beat the initial IoU by this
    fix_scale: bool = True  # perspective makes scale and depth near-degenerate; free scale only on request
    polish_widths: tuple = (320, 640)  # coarse-to-fine resolutions of the IoU polish
    polish_iou: bool = True  # derivative-free silhouette-IoU stage after the chamfer stage

    @staticmethod
    def from_json(d):
        c = RegConfig()
        for k, v in (d or {}).items():
            if not hasattr(c, k):
                raise InputError(f"unknown registration option {k!r}")
            setattr(c, k, tuple(v) if k in ("scale_bounds", "polish_widths") else v)
        return c


def silhouette(V, F, pose, K, width, height, face_tooth=None):
    """Binary silhouette (union of projected triangles).

    NOTE: cv2.fillPoly applies the even-odd rule across all polygons of one call, so overlapping triangles (front and back
    faces) cancel out. We therefore fill union-safely: per-tooth convex hulls when tooth labels exist (crowns are close to
    convex in projection; slight over-estimate at notches), else one fillConvexPoly per triangle.
    """
    uv, z, valid = project(V, pose, K)
    ok = valid[F].all(1)
    m = np.zeros((height, width), np.uint8)
    if not ok.any():
        return m
    F = F[ok]
    P = np.round(uv * 16).astype(np.int32)  # 4 fractional bits
    if face_tooth is not None and len(np.unique(face_tooth)) > 1:
        ft = np.asarray(face_tooth)[ok]
        for t in np.unique(ft):
            hull = cv2.convexHull(P[np.unique(F[ft == t])])
            cv2.fillConvexPoly(m, hull, 255, shift=4)
    else:
        for tri in P[F]:
            cv2.fillConvexPoly(m, tri, 255, shift=4)
    return m


def iou(a, b) -> float:
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def pose_from_correspondences(P3, P2, K, center, scale=1.0) -> Pose | None:
    """PnP from genuine 3D-2D correspondences (>=4, non-degenerate). Returns None if it fails."""
    P3, P2 = np.asarray(P3, float), np.asarray(P2, float)
    if len(P3) < 4 or np.linalg.matrix_rank(P3 - P3.mean(0), tol=1e-6) < 2:
        return None
    ok, rvec, tvec = cv2.solvePnP(P3, P2, K, None, flags=cv2.SOLVEPNP_EPNP if len(P3) > 4 else cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(P3, P2, K, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    R = Rotation.from_rotvec(rvec.ravel()).as_matrix()
    t = tvec.ravel() + R @ center  # solvePnP maps raw model coords; convert to pivot-centred form
    if t[2] <= 0:
        return None
    return Pose(rvec.ravel(), t, scale, np.asarray(center, float))


def mask_init(arch: Arch, mask, K, base: Pose) -> Pose:
    """Place the frontal pose so the model's projected bbox centre/size matches the mask bbox (translation+depth only)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        raise InputError("tooth mask is empty/too small")
    uv, _, valid = project(arch.vertices, base, K)
    w_model, w_mask = np.ptp(uv[:, 0]), np.ptp(xs)
    tz = base.t[2] * w_model / max(w_mask, 1)
    p = Pose(base.rvec, np.array([0, 0, tz]), base.scale, base.center)
    uv, _, _ = project(arch.vertices, p, K)
    d = np.array([(xs.min() + xs.max()) / 2 - (uv[:, 0].min() + uv[:, 0].max()) / 2,
                  (ys.min() + ys.max()) / 2 - (uv[:, 1].min() + uv[:, 1].max()) / 2])
    p.t = np.array([d[0] * tz / K[0, 0], d[1] * tz / K[1, 1], tz])
    return p


def _polish_iou(Ad, mask, K, pose, base, cfg, lm3, lm2, w, h, width=320):
    """Powell search maximising low-res silhouette IoU (+ landmark/scale penalties), within the same bounds."""
    from scipy.optimize import minimize
    f = min(width / w, 1.0)
    Ks = K.copy()
    Ks[:2] *= f
    ms = cv2.resize(mask, (int(w * f), int(h * f)), interpolation=cv2.INTER_NEAREST)
    sc = np.array([0.05, 0.05, 0.05, 5.0, 5.0, 20.0, 0.05])
    n = 6 if cfg.fix_scale else 7

    def mk(y):
        x = y * sc[:n]
        rv = (Rotation.from_rotvec(x[:3]) * Rotation.from_rotvec(pose.rvec)).as_rotvec()
        return Pose(rv, pose.t + x[3:6], pose.scale if cfg.fix_scale else pose.scale * np.exp(x[6]), pose.center)

    def cost(y):
        p = mk(y)
        rel = (Rotation.from_rotvec(p.rvec) * Rotation.from_rotvec(base.rvec).inv()).magnitude()
        if (np.degrees(rel) > cfg.max_rot_deg or p.t[2] < 0.4 * base.t[2] or p.t[2] > 2.5 * base.t[2]
                or not cfg.scale_bounds[0] <= p.scale <= cfg.scale_bounds[1]):
            return 1e3
        c = 1 - iou(silhouette(Ad.vertices, Ad.faces, p, Ks, ms.shape[1], ms.shape[0], Ad.face_tooth), ms)
        if len(lm3):
            uv, _, ok = project(lm3, p, K)
            c += cfg.w_landmark * np.mean(np.sum((uv - lm2) ** 2, 1)) ** 0.5 / (0.1 * w) if ok.all() else 1e3
        return c

    r = minimize(cost, np.zeros(n), method="Powell", options={"xtol": 1e-2, "ftol": 1e-4, "maxfev": 600})
    return mk(r.x) if r.fun < cost(np.zeros(n)) else pose


def register(arch: Arch, image_shape, K, *, landmarks=None, mask=None, cfg: RegConfig | None = None,
             init: Pose | None = None, vertices=None):
    """Estimate pose of `arch` for a photograph.

    landmarks: list of (fdi, (u, v)) -- model point = that tooth's centroid (genuine correspondence).
    mask: uint8 tooth mask or None. Needs landmarks and/or mask; otherwise raises InputError
    (manual alignment remains available). Returns (Pose, report dict).
    """
    cfg = cfg or RegConfig()
    h, w = image_shape[:2]
    V = arch.vertices if vertices is None else vertices
    lm = [(f, uv) for f, uv in (landmarks or []) if f in arch.teeth]
    if len(lm) != len(landmarks or []):
        raise InputError("a landmark refers to a tooth that is not in the model")
    if mask is None and len(lm) < 4:
        raise InputError("automatic alignment needs a tooth mask/contour or >= 4 tooth landmarks")
    notes = []
    base = init or initial_pose(arch, w, h, K)
    method = None
    if len(lm) >= 4:
        p0 = pose_from_correspondences([arch.teeth[f] for f, _ in lm], [uv for _, uv in lm], K, arch.center)
        if p0 is None:
            notes.append("PnP failed/degenerate" + ("; used mask-based initialisation" if mask is not None else ""))
        else:
            base, method = p0, "pnp"
    if method is None:
        if mask is None:
            raise InputError("could not initialise: degenerate landmark configuration and no mask")
        base = mask_init(arch, mask, K, base)
        method = "mask_bbox"
        if lm:
            notes.append(f"only {len(lm)} landmark(s): used as refinement residuals, not PnP")

    rng = np.random.default_rng(0)
    # decimate for speed; normals for back-face culling
    Ad = decimate(Arch(arch.name, arch.jaw, V, arch.faces, arch.face_tooth, arch.teeth), 6000)
    nrm = Ad.normals()
    sel = rng.choice(len(Ad.vertices), min(cfg.n_model_pts * 2, len(Ad.vertices)), replace=False)
    lm3 = np.array([arch.teeth[f] for f, _ in lm]) if lm else np.zeros((0, 3))
    lm2 = np.array([uv for _, uv in lm]) if lm else np.zeros((0, 2))
    if mask is not None:
        inv_dt = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        ey, ex = np.nonzero(edge)
        pick = rng.choice(len(ex), min(cfg.n_mask_pts, len(ex)), replace=False)
        mpts = np.c_[ex[pick], ey[pick]].astype(float)
        mask_area = float((mask > 0).sum())

    sc = np.array([0.1, 0.1, 0.1, 10.0, 10.0, 40.0, 0.1])  # variable scaling -> usable finite-difference steps

    def unpack(y):
        x = y * sc[: len(y)]
        s = base.scale if cfg.fix_scale else float(np.exp(x[6]))
        rv = (Rotation.from_rotvec(x[:3]) * Rotation.from_rotvec(base.rvec)).as_rotvec()
        return Pose(rv, base.t + x[3:6], s, base.center)

    def residuals(y):
        p = unpack(y)
        r = []
        if len(lm3):
            uv, _, ok = project(lm3, p, K)
            r.append(cfg.w_landmark * np.where(ok[:, None], uv - lm2, 1e3).ravel())
        if mask is not None:
            uv, z, ok = project(Ad.vertices[sel], p, K)
            n_cam = nrm[sel] @ p.R().T
            front = ok & (n_cam[:, 2] < 0.1)  # facing the camera
            # model -> mask: visible points outside the mask are penalised by distance to the mask
            ij = np.clip(np.round(uv).astype(int), [0, 0], [w - 1, h - 1])
            d = np.where(front, inv_dt[ij[:, 1], ij[:, 0]], 0.0)
            out_of_frame = front & ((uv[:, 0] < 0) | (uv[:, 0] > w) | (uv[:, 1] < 0) | (uv[:, 1] > h))
            d = np.where(out_of_frame, 200.0, d)
            r.append(cfg.w_model_to_mask * d[: cfg.n_model_pts] if len(d) >= cfg.n_model_pts
                     else np.pad(d, (0, cfg.n_model_pts - len(d))))
            # mask -> model: every mask boundary sample should have a visible model point nearby
            uv_all, _, ok_all = project(Ad.vertices, p, K)
            front_all = ok_all & ((nrm @ p.R().T)[:, 2] < 0.1)
            if front_all.sum() >= 3:
                dd, _ = cKDTree(uv_all[front_all]).query(mpts)
            else:
                dd = np.full(len(mpts), 200.0)
            r.append(cfg.w_mask_to_model * np.pad(dd, (0, cfg.n_mask_pts - len(dd))))
        if not cfg.fix_scale:
            r.append(np.array([cfg.w_scale_prior * y[6] * sc[6]]))
        return np.concatenate(r)

    n = 6 if cfg.fix_scale else 7
    rot = np.radians(cfg.max_rot_deg)
    lo = np.r_[[-rot] * 3, -300, -300, -base.t[2] * 0.6, np.log(cfg.scale_bounds[0])][:n] / sc[:n]
    hi = np.r_[[rot] * 3, 300, 300, base.t[2] * 1.5, np.log(cfg.scale_bounds[1])][:n] / sc[:n]
    x0 = np.zeros(n)

    def metrics(p):
        out = {}
        if len(lm3):
            uv, _, ok = project(lm3, p, K)
            out["landmark_rmse_px"] = float(np.sqrt(np.mean(np.sum((uv - lm2) ** 2, 1)))) if ok.all() else None
        if mask is not None:
            sil = silhouette(V, arch.faces, p, K, w, h, arch.face_tooth)
            out["silhouette_iou"] = iou(sil, mask)
            uv, _, ok = project(Ad.vertices[sel], p, K)
            ij = np.clip(np.round(uv).astype(int), [0, 0], [w - 1, h - 1])
            out["mean_model_to_mask_px"] = float(np.mean(inv_dt[ij[:, 1], ij[:, 0]]))
        return out

    m0 = metrics(base)
    status, sol = "ok", None
    try:
        sol = least_squares(residuals, x0, bounds=(lo, hi), loss=cfg.loss, f_scale=cfg.f_scale,
                            max_nfev=cfg.max_nfev, diff_step=0.02)
        pose = unpack(sol.x)
    except Exception as e:  # numerical failure: keep initial pose, say so
        pose, status = base, "failed"
        notes.append(f"optimizer error: {e}")
    if mask is not None and sol is not None and cfg.polish_iou:
        for pw in cfg.polish_widths:  # coarse then fine
            pose = _polish_iou(Ad, mask, K, pose, base, cfg, lm3, lm2, w, h, pw)
    m1 = metrics(pose)
    # safeguards: reject refinement that makes the primary metric worse
    key, better = ("silhouette_iou", lambda a, b: b > a + cfg.min_iou_gain) if mask is not None else \
        ("landmark_rmse_px", lambda a, b: b is not None and a is not None and b < a)
    lm0, lm1 = m0.get("landmark_rmse_px"), m1.get("landmark_rmse_px")
    lm_worse = lm0 is not None and (lm1 is None or lm1 > max(lm0 * 1.5, lm0 + 5.0))  # px
    if sol is not None and (lm_worse or not better(m0.get(key), m1.get(key))):
        pose, m1 = base, m0
        notes.append("refinement did not improve the fit (or worsened the landmark fit); the initial pose (from "
                     f"{method}) was kept")
    if sol is not None and sol.status == 0:
        notes.append("optimizer hit max_nfev")
    if mask is not None and m1.get("silhouette_iou", 1) < 0.5:
        status = "poor_match" if status == "ok" else status
        notes.append("low silhouette IoU: result is unreliable; adjust manually")
    return pose, {"nfev": int(sol.nfev) if sol is not None else 0, "optimizer_message": sol.message if sol is not None else None, "method": method, "status": status, "initial_metrics": m0, "metrics": m1,
                  "n_landmarks": len(lm), "mask_used": mask is not None, "notes": notes,
                  "config": cfg.__dict__ | {"scale_bounds": list(cfg.scale_bounds), "polish_widths": list(cfg.polish_widths)}}
