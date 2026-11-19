"""
dental_overlay.py  -  Phase 3: overlay a 3D tooth arch on a FRONTAL 2D photo.

What this does (and why the old version never lined up)
-------------------------------------------------------
The old tool flattened the arch along the wrong axis, giving a top-down
(occlusal) U-shape.  A frontal photo shows the arch from the FRONT, so no
2D shift/rotate/scale could ever match it.

This version:
  1. Works out the patient's anatomical axes from the FDI tooth numbers
     (left-right, front-back, up-down) so it does NOT depend on whatever
     coordinate system the scanner used.
  2. Renders the arch as seen from the FRONT, as a shaded 3D model with
     proper depth ordering and tooth outlines, with a perspective camera.
  3. Fits the camera pose (scale, roll, shift, perspective, small pitch/yaw)
     from the teeth you click in the photo.
  4. Optionally snaps each tooth onto its clicked position (per-tooth
     adjustment) - needed because the 3D scan and the photo are normally
     different people, so teeth never match exactly.
  5. Lets you fine-tune live with sliders (only re-renders when something
     changes, so it stays smooth).

Only needs: numpy, opencv-python, trimesh   (NOT open3d)
"""

import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np
import cv2

try:
    import trimesh
except ImportError:  # pragma: no cover
    raise SystemExit("trimesh is required:  pip install trimesh")


# --------------------------------------------------------------------------
# Settings you may want to edit
# --------------------------------------------------------------------------
DEFAULT_CASE = (r"C:\Users\utkar\Downloads\DENTAL"
                r"\VITA-3D-Master-Shade-System-for-Dental-Simulators"
                r"\processed_part1_upper\M7RTPNPC")
DEFAULT_PHOTO = "patient_photo.jpg"

MAX_DISPLAY = 900            # longest side (px) of on-screen windows
TARGET_FACES_PER_TOOTH = 1200  # mesh is simplified to this for fast drawing
MAX_CLICK_INDEX = 6          # only ask you to click teeth 1..6 of each side
DEFAULT_PERSP = 35           # default perspective strength (0 = none)
SHIFT_BITS = 4               # sub-pixel precision for polygon drawing
OUTLINE_COLOR = (0, 255, 255)  # BGR (yellow)

TOOTH_NAMES = {1: "central incisor", 2: "lateral incisor", 3: "canine",
               4: "1st premolar", 5: "2nd premolar", 6: "1st molar",
               7: "2nd molar", 8: "3rd molar"}
QUADRANT_NAMES = {1: "upper RIGHT", 2: "upper LEFT",
                  3: "lower LEFT", 4: "lower RIGHT"}
BASE_COLORS_BGR = [(120, 215, 80), (200, 200, 60)]   # alternate per tooth


# --------------------------------------------------------------------------
# Loading meshes
# --------------------------------------------------------------------------
def _check_not_lfs_pointer(path):
    if os.path.getsize(path) < 2048:
        with open(path, "rb") as f:
            head = f.read(200)
        if head.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{path} is a Git-LFS pointer, not real data. "
                f"Run 'git lfs pull' inside the cloned repository.")


def _cluster(V, F, cell):
    """Vertex-clustering mesh simplification (pure numpy)."""
    q = np.floor((V - V.min(axis=0)) / cell).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    V2 = np.stack([np.bincount(inv, weights=V[:, k], minlength=n) / cnt
                   for k in range(3)], axis=1)
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[ok]
    if len(F2):
        key = np.sort(F2, axis=1)
        _, ui = np.unique(key, axis=0, return_index=True)
        F2 = F2[np.sort(ui)]
    return V2, F2


def decimate(V, F, target):
    if len(F) <= target:
        return V, F
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    cell = diag / 150.0
    best = (V, F)
    for _ in range(30):
        V2, F2 = _cluster(V, F, cell)
        if len(F2) < 40:
            break
        best = (V2, F2)
        if len(F2) <= target:
            break
        cell *= 1.18
    return best


def load_case(case_dir, target_faces=TARGET_FACES_PER_TOOTH):
    """Load every tooth_<FDI>.obj in a case folder -> {fdi: {'V','F'}}."""
    files = sorted(glob.glob(os.path.join(case_dir, "tooth_*.obj")))
    if not files:
        raise FileNotFoundError(f"No tooth_*.obj files in: {case_dir}")
    teeth = {}
    for path in files:
        m = re.search(r"tooth_(\d+)\.obj$", os.path.basename(path))
        if not m:
            continue
        fdi = int(m.group(1))
        if not (11 <= fdi <= 48 and 1 <= fdi % 10 <= 8):
            continue
        _check_not_lfs_pointer(path)
        try:
            mesh = trimesh.load(path, force="mesh", process=False)
        except Exception as exc:  # noqa
            print(f"  warning: could not read tooth {fdi} ({exc}) - skipped")
            continue
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)
        if len(V) == 0 or len(F) == 0:
            print(f"  warning: tooth {fdi} has no geometry - skipped")
            continue
        V, F = decimate(V, F, target_faces)
        teeth[fdi] = {"V": V, "F": F}
    if len(teeth) < 4:
        raise ValueError(f"Only {len(teeth)} usable teeth found in {case_dir}")
    print(f"Loaded {len(teeth)} teeth: {sorted(teeth)}")
    return teeth


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def face_geometry(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    a2 = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(a2[:, None], 1e-12)
    return n, a2 / 2.0, (a + b + c) / 3.0


def orient_outward(V, F):
    """Make triangle winding point outward (robust to inverted meshes)."""
    n, area, cen = face_geometry(V, F)
    tc = (area[:, None] * cen).sum(0) / max(area.sum(), 1e-12)
    score = (area * ((cen - tc) * n).sum(1)).sum()
    return F[:, [0, 2, 1]] if score < 0 else F


def prepare_teeth(teeth):
    """Per tooth: area-weighted centroid + outward normals sum (raw coords)."""
    for fdi, t in teeth.items():
        t["F"] = orient_outward(t["V"], t["F"])
        n, area, cen = face_geometry(t["V"], t["F"])
        t["centroid"] = (area[:, None] * cen).sum(0) / area.sum()
        t["nsum"] = (area[:, None] * n).sum(0)
        t["area"] = float(area.sum())
    return teeth


def anatomical_frame(teeth):
    """
    Patient axes from FDI numbers (independent of scanner orientation):
      L = patient's LEFT   (quadrants 2,3 minus quadrants 1,4)  -> photo right
      A = anterior (front, towards the lips)
      D = down  = L x A   (right-handed, so no mirror image)
    Camera frame:  u = along L (photo x), v = along D (photo y),
                   w = -A (away from the camera, i.e. further back in mouth)
    """
    cents = {f: t["centroid"] for f, t in teeth.items()}
    right = [c for f, c in cents.items() if f // 10 in (1, 4)]
    left = [c for f, c in cents.items() if f // 10 in (2, 3)]
    if not right or not left:
        raise ValueError("Need teeth from both the patient's right and left side")
    L = np.mean(left, 0) - np.mean(right, 0)
    L /= np.linalg.norm(L)
    ant = [c for f, c in cents.items() if f % 10 in (1, 2)]
    if not ant:
        mn = min(f % 10 for f in cents)
        ant = [c for f, c in cents.items() if f % 10 == mn]
    M = np.mean(list(cents.values()), 0)
    I = np.mean(ant, 0)
    A = I - M
    A = A - A.dot(L) * L
    A /= np.linalg.norm(A)
    D = np.cross(L, A)
    R = np.stack([L, D, -A])            # rows = camera axes, det = +1

    # check: crowns (open surfaces) should point DOWN (+D) for an upper arch,
    # UP for a lower arch.  Sum of outward normals of an open crown points
    # towards its biting surface.  If this strongly disagrees, flip vertically.
    upper = any(f // 10 in (1, 2) for f in teeth)
    nsum = sum(t["nsum"] for t in teeth.values())
    ratio = float(np.linalg.norm(nsum) / max(sum(t["area"] for t in teeth.values()), 1e-12))
    occ = nsum / max(np.linalg.norm(nsum), 1e-12)
    agree = float(occ.dot(D if upper else -D))
    auto_flip_v = bool(ratio > 0.2 and agree < -0.4)
    if auto_flip_v:
        print("  NOTE: scan orientation looked upside-down relative to its labels - "
              "corrected automatically (press 'v' if it looks wrong).")
    elif ratio > 0.2 and agree < 0.0:
        print("  NOTE: orientation check is unsure. If the overlay looks "
              "upside-down press 'v'; if mirrored press 'h'.")

    # model-unit sanity (expect millimetres): centrals are ~8.5 mm apart (upper)
    unit_scale = 1.0
    if 11 in cents and 21 in cents:
        d = np.linalg.norm(cents[11] - cents[21])
        if not (4.0 <= d <= 14.0) and d > 1e-9:
            unit_scale = 8.8 / d
            print(f"  NOTE: model units don't look like mm; rescaling x{unit_scale:.3f}")
    return {"R": R, "O": I, "unit_scale": unit_scale, "upper": upper,
            "auto_flip_v": auto_flip_v}


def to_camera_frame(teeth, frame, flip_u=False, flip_v=False):
    R = frame["R"].copy()
    if flip_u:
        R[0] *= -1
    if flip_v:
        R[1] *= -1
    out = {}
    for fdi, t in teeth.items():
        P = ((t["V"] - frame["O"]) @ R.T) * frame["unit_scale"]
        F = orient_outward(P, t["F"])
        n, area, cen = face_geometry(P, F)
        wv = area * np.maximum(0.0, -n[:, 2])      # frontal-facing surface
        if wv.sum() > 1e-6 * area.sum():
            ref = (wv[:, None] * cen).sum(0) / wv.sum()
        else:
            ref = (area[:, None] * cen).sum(0) / area.sum()
        out[fdi] = {"P": P, "F": F, "n": n, "ref": ref}
    return out


# --------------------------------------------------------------------------
# Camera model + fitting
# --------------------------------------------------------------------------
def rot_pitch_yaw(pitch_deg, yaw_deg):
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)],
                   [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]])
    return Ry @ Rx


def plane_coords(P, q, pitch, yaw):
    """3D (u,v,w) -> perspective image-plane coords (complex) + depth."""
    Pr = P @ rot_pitch_yaw(pitch, yaw).T
    k = 1.0 / np.maximum(0.25, 1.0 + q * Pr[:, 2])
    return (Pr[:, 0] + 1j * Pr[:, 1]) * k, Pr[:, 2]


def similarity_fit(z, w):
    """Least-squares  w ~ a*z + b  (a complex = scale*rotation)."""
    zm, wm = z.mean(), w.mean()
    zc, wc = z - zm, w - wm
    den = float((np.abs(zc) ** 2).sum())
    if den < 1e-12:
        raise ValueError("clicked/model points are all in the same place")
    a = (np.conj(zc) * wc).sum() / den
    b = wm - a * zm
    rms = float(np.sqrt((np.abs(a * z + b - w) ** 2).mean()))
    return a, b, rms


def fit_pose(cam, clicks, extended=True):
    fd = sorted(set(cam) & set(clicks))
    if len(fd) < 4:
        raise ValueError(f"Need at least 4 clicked teeth that exist in the "
                         f"3D model; got {len(fd)}")
    ref = np.array([cam[f]["ref"] for f in fd])
    w = np.array([complex(*clicks[f]) for f in fd])
    if extended and len(fd) >= 6:
        pgrid, pitchg, yawg = range(0, 81, 10), range(-20, 21, 5), range(-20, 21, 5)
    else:
        pgrid, pitchg, yawg = [DEFAULT_PERSP], [0], [0]
    best = None
    for pv in pgrid:
        for pitch in pitchg:
            for yaw in yawg:
                z, _ = plane_coords(ref, pv / 1e4, pitch, yaw)
                a, b, rms = similarity_fit(z, w)
                s = abs(a)
                score = rms / s + 0.02 * (abs(pitch) + abs(yaw)) + 0.003 * pv
                if best is None or score < best[0]:
                    best = (score, pv, pitch, yaw, a, b, rms, s)
    _, pv, pitch, yaw, a, b, rms, s = best
    return {"a0": a, "b0": b, "persp": pv, "q": pv / 1e4, "pitch": pitch,
            "yaw": yaw, "rms_px": rms, "rms_mm": rms / s, "scale_px_per_mm": s,
            "n_used": len(fd), "used": fd}


def snap_offsets(cam, clicks, pose, pv=None, pitch=None, yaw=None):
    """Per-tooth shift (in image-plane units) putting each tooth on its click."""
    q = pose["q"] if pv is None else pv / 1e4
    pitch = pose["pitch"] if pitch is None else pitch
    yaw = pose["yaw"] if yaw is None else yaw
    offs, uvals = {}, {}
    for f in cam:
        uvals[f] = cam[f]["ref"][0]
    for f in cam:
        if f in clicks:
            z, _ = plane_coords(cam[f]["ref"][None, :], q, pitch, yaw)
            desired = (complex(*clicks[f]) - pose["b0"]) / pose["a0"]
            offs[f] = desired - z[0]
    if not offs:
        return {f: 0j for f in cam}
    for f in cam:
        if f not in offs:                       # un-clicked tooth: borrow nearest
            nearest = min(offs, key=lambda g: abs(uvals[g] - uvals[f]))
            offs[f] = offs[nearest]
    return offs


# --------------------------------------------------------------------------
# Renderer (shaded, depth-sorted, with outlines)
# --------------------------------------------------------------------------
class ArchRenderer:
    def __init__(self, cam):
        self.fdis = sorted(cam)
        self.label_id = {f: i + 1 for i, f in enumerate(self.fdis)}
        Vs, Fs, Ns, tv, tf, off = [], [], [], [], [], 0
        for f in self.fdis:
            t = cam[f]
            Vs.append(t["P"])
            Fs.append(t["F"] + off)
            Ns.append(t["n"])
            tv.append(np.full(len(t["P"]), self.label_id[f]))
            tf.append(np.full(len(t["F"]), self.label_id[f]))
            off += len(t["P"])
        self.V, self.F, self.N = np.vstack(Vs), np.vstack(Fs), np.vstack(Ns)
        self.tv, self.tf = np.concatenate(tv), np.concatenate(tf)
        self.refs = {f: cam[f]["ref"] for f in self.fdis}
        base = np.array([BASE_COLORS_BGR[i % 2] for i in range(len(self.fdis))],
                        dtype=np.float64)
        self.base_face = np.vstack([[0, 0, 0], base])[self.tf]

    @staticmethod
    def _map(P, pose, off=None):
        Pr = P @ rot_pitch_yaw(pose["pitch"], pose["yaw"]).T
        k = 1.0 / np.maximum(0.25, 1.0 + pose["q"] * Pr[:, 2])
        plane = (Pr[:, 0] + 1j * Pr[:, 1]) * k
        if off is not None:
            plane = plane + off
        z = pose["a0"] * plane + pose["b0"]
        g = pose["g_scale"] * np.exp(1j * math.radians(pose["g_roll"]))
        z = (z - pose["center"]) * g + pose["center"] + complex(pose["g_dx"], pose["g_dy"])
        return z, Pr

    def render(self, pose, out_w, out_h, ps):
        offs = pose.get("offs")
        off_arr = None
        if offs is not None:
            arr = np.zeros(len(self.fdis) + 1, dtype=complex)
            for f in self.fdis:
                arr[self.label_id[f]] = offs.get(f, 0j)
            off_arr = arr
        z, Pr = self._map(self.V, pose, off_arr[self.tv] if off_arr is not None else None)
        xy = np.clip(np.stack([z.real, z.imag], axis=1) * ps, -1e6, 1e6)
        depth = Pr[:, 2]
        R = rot_pitch_yaw(pose["pitch"], pose["yaw"])
        nz = (self.N @ R.T)[:, 2]
        lam = np.clip(-nz, 0.0, 1.0)
        vis = np.nonzero(nz < 0.05)[0]
        zf = depth[self.F[vis]].mean(axis=1)
        order = vis[np.argsort(-zf)]                    # far first
        pts = np.round(xy[self.F[order]] * (1 << SHIFT_BITS)).astype(np.int32)
        shade = np.clip(self.base_face[order] * (0.45 + 0.55 * lam[order])[:, None],
                        0, 255).astype(np.uint8)
        cols = [tuple(c) for c in shade.tolist()]
        ids = self.tf[order].tolist()
        img = np.zeros((out_h, out_w, 3), np.uint8)
        lab = np.zeros((out_h, out_w), np.uint8)
        fill = cv2.fillConvexPoly
        for i in range(len(order)):
            fill(img, pts[i], cols[i], cv2.LINE_8, SHIFT_BITS)
            fill(lab, pts[i], ids[i], cv2.LINE_8, SHIFT_BITS)
        ref_arr = np.array([self.refs[f] for f in self.fdis])
        roff = None
        if off_arr is not None:
            roff = np.array([off_arr[self.label_id[f]] for f in self.fdis])
        rz, _ = self._map(ref_arr, pose, roff)
        ref_xy = {f: (rz[i].real * ps, rz[i].imag * ps) for i, f in enumerate(self.fdis)}
        return {"shade": img, "lab": lab, "ref_xy": ref_xy}


def outline_mask(lab):
    e = np.zeros(lab.shape, bool)
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    e[1:, :] |= lab[1:, :] != lab[:-1, :]
    return e


def compose(photo, rend, opacity=0.55, outlines=True, labels=True):
    lab, shade = rend["lab"], rend["shade"]
    mask = lab > 0
    out = photo.copy()
    if mask.any():
        blend = cv2.addWeighted(photo, 1.0 - opacity, shade, opacity, 0)
        out[mask] = blend[mask]
    if outlines:
        e = outline_mask(lab)
        k = max(1, int(round(photo.shape[1] / 700.0)))
        if k > 1:
            e = cv2.dilate(e.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        out[e] = OUTLINE_COLOR
    if labels:
        fs = max(0.4, photo.shape[1] / 1400.0)
        th = max(1, int(round(photo.shape[1] / 700.0)))
        H, W = photo.shape[:2]
        for f, (x, y) in rend["ref_xy"].items():
            if f % 10 <= MAX_CLICK_INDEX and 0 <= x < W and 0 <= y < H:
                cv2.putText(out, str(f), (int(x) - 10, int(y) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.putText(out, str(f), (int(x) - 10, int(y) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# Session = everything that depends on orientation flags
# --------------------------------------------------------------------------
class Session:
    def __init__(self, teeth, frame, photo, clicks, case_dir=""):
        self.teeth, self.frame, self.photo = teeth, frame, photo
        self.clicks, self.case_dir = clicks, case_dir
        self.flip_u = False
        self.flip_v = bool(frame.get("auto_flip_v", False))
        self.snap = True
        self.outline = True
        self.labels = True
        self.compare = False
        self.center = complex(np.mean([c[0] for c in clicks.values()]),
                              np.mean([c[1] for c in clicks.values()]))
        self.auto_orient()

    def auto_orient(self):
        """Pick mirrored/not by which one fits the clicks better."""
        trials = []
        for fu in (False, True):
            cam = to_camera_frame(self.teeth, self.frame, fu, self.flip_v)
            pose = fit_pose(cam, self.clicks, extended=False)
            trials.append((pose["rms_mm"], fu))
        (r0, _), (r1, _) = trials
        self.flip_u = (r1 < 0.6 * r0)
        if self.flip_u:
            print("  NOTE: left/right looked mirrored - corrected automatically.")
        self.rebuild()

    def rebuild(self):
        self.cam = to_camera_frame(self.teeth, self.frame, self.flip_u, self.flip_v)
        self.pose = fit_pose(self.cam, self.clicks)
        self.renderer = ArchRenderer(self.cam)

    def pose_for(self, g_scale=1.0, g_roll=0.0, g_dx=0.0, g_dy=0.0,
                 pv=None, pitch=None, yaw=None):
        p = dict(self.pose)
        if pv is not None:
            p["q"], p["persp"] = pv / 1e4, pv
        if pitch is not None:
            p["pitch"] = pitch
        if yaw is not None:
            p["yaw"] = yaw
        p.update(g_scale=g_scale, g_roll=g_roll, g_dx=g_dx, g_dy=g_dy, center=self.center)
        p["offs"] = (snap_offsets(self.cam, self.clicks, self.pose,
                                  p["persp"], p["pitch"], p["yaw"])
                     if self.snap else None)
        return p

    def render_full(self, pose, opacity):
        H, W = self.photo.shape[:2]
        rend = self.renderer.render(pose, W, H, 1.0)
        return rend, compose(self.photo, rend, opacity, self.outline, self.labels)


# --------------------------------------------------------------------------
# GUI: clicking + fine-tune
# --------------------------------------------------------------------------
def click_order(fdis):
    right = sorted([f for f in fdis if f // 10 in (1, 4) and f % 10 <= MAX_CLICK_INDEX],
                   key=lambda f: -(f % 10))
    left = sorted([f for f in fdis if f // 10 in (2, 3) and f % 10 <= MAX_CLICK_INDEX],
                  key=lambda f: f % 10)
    return right + left


def load_photo(path):
    img = cv2.imread(path)
    if img is None or img.size == 0:
        raise FileNotFoundError(f"Could not read the photo: {path} "
                                f"(missing, empty or corrupted file)")
    return img


def collect_clicks(photo, fdis):
    H, W = photo.shape[:2]
    ps = min(1.0, MAX_DISPLAY / W, MAX_DISPLAY / H)
    small = cv2.resize(photo, (int(W * ps), int(H * ps)), interpolation=cv2.INTER_AREA)
    order = click_order(fdis)
    st = {"i": 0, "hist": [], "clicks": {}}
    win = "Click the teeth in order"

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and st["i"] < len(order):
            f = order[st["i"]]
            st["clicks"][f] = (x / ps, y / ps)
            st["hist"].append(("c", f))
            st["i"] += 1

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    print("\nClick the CENTRE of each tooth's visible crown, in the order shown.")
    print("  n = skip (tooth not visible)   u = undo   Enter = finish\n")
    while True:
        fr = small.copy()
        for f, (x, y) in st["clicks"].items():
            p = (int(x * ps), int(y * ps))
            cv2.circle(fr, p, 5, (0, 0, 255), -1)
            cv2.putText(fr, str(f), (p[0] + 7, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2, cv2.LINE_AA)
        if st["i"] < len(order):
            f = order[st["i"]]
            side = "LEFT" if f // 10 in (1, 4) else "RIGHT"
            msg = (f"Click tooth {f}: {QUADRANT_NAMES[f // 10]} "
                   f"{TOOTH_NAMES[f % 10]}  (photo {side} side)")
        else:
            msg = "All done - press Enter"
        cv2.rectangle(fr, (0, 0), (fr.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(fr, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, fr)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, ord("q")):
            break
        if k == ord("n") and st["i"] < len(order):
            st["hist"].append(("n", order[st["i"]]))
            st["i"] += 1
        if k == ord("u") and st["hist"]:
            kind, f = st["hist"].pop()
            st["i"] -= 1
            if kind == "c":
                st["clicks"].pop(f, None)
    cv2.destroyWindow(win)
    return st["clicks"]


def save_results(sess, pose, opacity, tag="final"):
    rend, out = sess.render_full(pose, opacity)
    cv2.imwrite(f"{tag}_overlay.png", out)
    mask = rend["lab"] > 0
    e = outline_mask(rend["lab"])
    rgba = np.dstack([rend["shade"], (mask * int(255 * opacity)).astype(np.uint8)])
    rgba[e] = (*OUTLINE_COLOR, 255)
    cv2.imwrite(f"{tag}_overlay_rgba.png", rgba)
    info = {
        "case_dir": sess.case_dir,
        "photo_size": [int(sess.photo.shape[1]), int(sess.photo.shape[0])],
        "clicks": {str(k): list(v) for k, v in sess.clicks.items()},
        "flip_u": sess.flip_u, "flip_v": sess.flip_v, "snap": sess.snap,
        "pose": {"a0": [pose["a0"].real, pose["a0"].imag],
                 "b0": [pose["b0"].real, pose["b0"].imag],
                 "perspective": pose["persp"], "pitch": pose["pitch"], "yaw": pose["yaw"],
                 "g_scale": pose["g_scale"], "g_roll": pose["g_roll"],
                 "g_dx": pose["g_dx"], "g_dy": pose["g_dy"]},
        "opacity": opacity,
    }
    with open(f"{tag}_pose.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved {tag}_overlay.png, {tag}_overlay_rgba.png, {tag}_pose.json")


def run_gui(sess):
    H, W = sess.photo.shape[:2]
    ps = min(1.0, MAX_DISPLAY / W, MAX_DISPLAY / H)
    sw, sh = int(W * ps), int(H * ps)
    small = cv2.resize(sess.photo, (sw, sh), interpolation=cv2.INTER_AREA)
    win = "Fine-tune"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    noop = lambda v: None
    bars = [("Scale %", 100, 200), ("Move X", 200, 400), ("Move Y", 200, 400),
            ("Roll", 30, 60), ("Perspective", DEFAULT_PERSP, 100),
            ("Pitch", 30, 60), ("Yaw", 30, 60), ("Opacity %", 55, 100)]

    def reset():
        init = {"Scale %": 100, "Move X": 200, "Move Y": 200, "Roll": 30,
                "Perspective": int(sess.pose["persp"]),
                "Pitch": 30 + int(sess.pose["pitch"]),
                "Yaw": 30 + int(sess.pose["yaw"]), "Opacity %": 55}
        for n, _, mx in bars:
            cv2.setTrackbarPos(n, win, int(np.clip(init[n], 0, mx)))

    for n, v, mx in bars:
        cv2.createTrackbar(n, win, v, mx, noop)
    reset()

    print("\nKeys:  t snap-to-clicks   o outlines   l labels   c compare with photo")
    print("       h mirror left/right   v flip up/down   r reset   s save   q quit")
    print(f"Fit: {sess.pose['n_used']} teeth, residual {sess.pose['rms_mm']:.2f} mm "
          f"(perspective {sess.pose['persp']}, pitch {sess.pose['pitch']}, "
          f"yaw {sess.pose['yaw']})")

    last, out = None, small
    while True:
        v = {n: cv2.getTrackbarPos(n, win) for n, _, _ in bars}
        key = (tuple(v.values()), sess.snap, sess.outline, sess.labels,
               sess.compare, sess.flip_u, sess.flip_v)
        if key != last:
            pose = sess.pose_for(max(0.1, v["Scale %"] / 100.0), v["Roll"] - 30,
                                 (v["Move X"] - 200) / ps, (v["Move Y"] - 200) / ps,
                                 v["Perspective"], v["Pitch"] - 30, v["Yaw"] - 30)
            if sess.compare:
                out = small.copy()
            else:
                rend = sess.renderer.render(pose, sw, sh, ps)
                out = compose(small, rend, v["Opacity %"] / 100.0,
                              sess.outline, sess.labels)
            hud = (f"snap:{'ON' if sess.snap else 'off'}  "
                   f"t snap | o outline | l labels | c compare | h mirror | v flip | r reset | s save | q quit")
            cv2.putText(out, hud, (8, sh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, hud, (8, sh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            last = key
        cv2.imshow(win, out)
        k = cv2.waitKey(25) & 0xFF
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        if k in (ord("q"), 27):
            break
        elif k == ord("t"):
            sess.snap = not sess.snap
        elif k == ord("o"):
            sess.outline = not sess.outline
        elif k == ord("l"):
            sess.labels = not sess.labels
        elif k == ord("c"):
            sess.compare = not sess.compare
        elif k in (ord("h"), ord("v")):
            if k == ord("h"):
                sess.flip_u = not sess.flip_u
            else:
                sess.flip_v = not sess.flip_v
            sess.rebuild()
            reset()
            last = None
        elif k == ord("r"):
            reset()
        elif k == ord("s"):
            pose = sess.pose_for(max(0.1, v["Scale %"] / 100.0), v["Roll"] - 30,
                                 (v["Move X"] - 200) / ps, (v["Move Y"] - 200) / ps,
                                 v["Perspective"], v["Pitch"] - 30, v["Yaw"] - 30)
            save_results(sess, pose, v["Opacity %"] / 100.0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--case", default=DEFAULT_CASE, help="folder with tooth_*.obj")
    ap.add_argument("--photo", default=DEFAULT_PHOTO, help="frontal teeth photo")
    args = ap.parse_args()

    print("Loading 3D teeth (a few seconds) ...")
    teeth = prepare_teeth(load_case(args.case))
    frame = anatomical_frame(teeth)
    photo = load_photo(args.photo)
    print(f"Photo: {photo.shape[1]}x{photo.shape[0]} px")

    clicks = collect_clicks(photo, list(teeth))
    clicks = {f: p for f, p in clicks.items() if f in teeth}
    if len(clicks) < 4:
        raise SystemExit(f"Only {len(clicks)} teeth clicked - need at least 4 "
                         f"(more is better). Run again.")
    sess = Session(teeth, frame, photo, clicks, args.case)
    run_gui(sess)


if __name__ == "__main__":
    main()