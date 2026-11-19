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
VERSION = "2026-11-13 (upper + lower jaw)"

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


def load_case(case_dir, target_faces=TARGET_FACES_PER_TOOTH, max_index=8, quiet=False):
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
        if not (11 <= fdi <= 48 and 1 <= fdi % 10 <= max_index):
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
    if not quiet:
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


def anatomical_frame(teeth, quiet=False):
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
    if quiet:
        pass
    elif auto_flip_v:
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
            if not quiet:
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
        cen3 = (area[:, None] * cen).sum(0) / area.sum()
        out[fdi] = {"P": P, "F": F, "n": n, "ref": ref, "centroid": cen3}
    # Facing angle of each tooth = direction of the arch curve at that tooth.
    # Walk along the arch from photo-left to photo-right; the outward normal of the
    # curve is (t_w, -t_u).  phi = 0 means the tooth faces straight at the camera.
    right = sorted([f for f in out if f // 10 in (1, 4)], key=lambda f: -(f % 10))
    left = sorted([f for f in out if f // 10 in (2, 3)], key=lambda f: f % 10)
    chain = right + left                      # anatomical order around the arch
    if len(chain) > 1 and out[chain[0]]["centroid"][0] > out[chain[-1]]["centroid"][0]:
        chain.reverse()
    for i, f in enumerate(chain):
        a = out[chain[max(i - 1, 0)]]["centroid"]
        b = out[chain[min(i + 1, len(chain) - 1)]]["centroid"]
        tu, tw = b[0] - a[0], b[2] - a[2]
        nrm = math.hypot(tu, tw)
        out[f]["phi"] = float(math.atan2(tw / nrm, tu / nrm)) if nrm > 1e-9 else 0.0
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


def fit_pose(cam, clicks, extended=True, min_points=4):
    fd = sorted(set(cam) & set(clicks))
    if len(fd) < min_points:
        raise ValueError(f"Need at least {min_points} clicked teeth that exist in the "
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
        self.phi = np.zeros(len(self.fdis) + 1)
        self.ref_by_id = np.zeros((len(self.fdis) + 1, 3))
        for f in self.fdis:
            self.phi[self.label_id[f]] = cam[f]["phi"]
            self.ref_by_id[self.label_id[f]] = cam[f]["ref"]
        base = np.array([BASE_COLORS_BGR[i % 2] for i in range(len(self.fdis))],
                        dtype=np.float64)
        self.base_face = np.vstack([[0, 0, 0], base])[self.tf]

    @staticmethod
    def _global(plane, pose):
        z = pose["a0"] * plane + pose["b0"]
        g = pose["g_scale"] * np.exp(1j * math.radians(pose["g_roll"]))
        return (z - pose["center"]) * g + pose["center"] + complex(pose["g_dx"], pose["g_dy"])

    def render(self, pose, out_w, out_h, ps):
        n = len(self.fdis)
        offs, scales = pose.get("offs"), pose.get("scales")
        off_arr = np.zeros(n + 1, dtype=complex)
        sx_arr = np.full(n + 1, float(pose.get("tw", 1.0)))     # tooth width multiplier
        sy_arr = np.full(n + 1, float(pose.get("th", 1.0)))     # tooth height multiplier
        for f in self.fdis:
            i = self.label_id[f]
            if offs:
                off_arr[i] = offs.get(f, 0j)
            if scales and f in scales:
                sx_arr[i] *= scales[f][0]
                sy_arr[i] *= scales[f][1]
        R = rot_pitch_yaw(pose["pitch"], pose["yaw"])
        ref_arr = np.array([self.refs[f] for f in self.fdis])
        rz0, _ = plane_coords(ref_arr, pose["q"], pose["pitch"], pose["yaw"])
        rp = np.zeros(n + 1, dtype=complex)
        rp[1:] = rz0
        V, N = self.V, self.N
        unf = float(pose.get("unfold", 0.0))
        if unf > 0:                       # turn every tooth about its own vertical axis towards the camera
            th_v = unf * self.phi[self.tv]
            c, sn = np.cos(th_v), np.sin(th_v)
            rv = self.ref_by_id[self.tv]
            du, dw = V[:, 0] - rv[:, 0], V[:, 2] - rv[:, 2]
            V = np.stack([rv[:, 0] + du * c + dw * sn, V[:, 1],
                          rv[:, 2] - du * sn + dw * c], axis=1)
            th_f = unf * self.phi[self.tf]
            cf_, sf_ = np.cos(th_f), np.sin(th_f)
            N = np.stack([N[:, 0] * cf_ + N[:, 2] * sf_, N[:, 1],
                          -N[:, 0] * sf_ + N[:, 2] * cf_], axis=1)
        Pr = V @ R.T
        k = 1.0 / np.maximum(0.25, 1.0 + pose["q"] * Pr[:, 2])
        plane = (Pr[:, 0] + 1j * Pr[:, 1]) * k
        d = plane - rp[self.tv]
        plane = (rp[self.tv] + d.real * sx_arr[self.tv] + 1j * d.imag * sy_arr[self.tv]
                 + off_arr[self.tv])
        z = self._global(plane, pose)
        xy = np.clip(np.stack([z.real, z.imag], axis=1) * ps, -1e6, 1e6)
        depth = Pr[:, 2]
        nz = (N @ R.T)[:, 2]
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
        rz = self._global(rp[1:] + off_arr[1:], pose)
        ref_xy = {f: (rz[i].real * ps, rz[i].imag * ps) for i, f in enumerate(self.fdis)}
        return {"shade": img, "lab": lab, "ref_xy": ref_xy, "label_id": self.label_id}


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
        counts = np.bincount(lab.ravel(), minlength=256)
        ids = rend.get("label_id", {})
        pos = [counts[i] for i in ids.values() if counts[i] > 0]
        med = float(np.median(pos)) if pos else 0.0
        fs = max(0.4, photo.shape[1] / 1400.0)
        th = max(1, int(round(photo.shape[1] / 700.0)))
        H, W = photo.shape[:2]
        for f, (x, y) in rend["ref_xy"].items():
            big_enough = (not ids) or counts[ids.get(f, 0)] >= 0.3 * med
            if f % 10 <= MAX_CLICK_INDEX and big_enough and 0 <= x < W and 0 <= y < H:
                cv2.putText(out, str(f), (int(x) - 10, int(y) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.putText(out, str(f), (int(x) - 10, int(y) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# Session = everything that depends on orientation flags
# --------------------------------------------------------------------------
class Session:
    def __init__(self, teeth, frame, photo, clicks, case_dir="",
                 flip_u=None, flip_v=None, contours=None):
        self.teeth, self.frame, self.photo = teeth, frame, photo
        self.clicks, self.case_dir = dict(clicks), case_dir
        self.contours = contours
        self.flip_v = bool(frame.get("auto_flip_v", False)) if flip_v is None else flip_v
        self.snap, self.outline, self.labels, self.compare = True, True, True, False
        self.scales, self.scales_on = None, False
        self.bias = {}
        self.center = self._center()
        if flip_u is None:
            self.flip_u = False
            self.auto_orient()
        else:
            self.flip_u = flip_u
            self.rebuild()

    def _center(self):
        return complex(np.mean([c[0] for c in self.clicks.values()]),
                       np.mean([c[1] for c in self.clicks.values()]))

    def auto_orient(self):
        """Pick mirrored/not by which one fits the clicks better."""
        trials = []
        for fu in (False, True):
            cam = to_camera_frame(self.teeth, self.frame, fu, self.flip_v)
            trials.append(fit_pose(cam, self.clicks, extended=False)["rms_mm"])
        self.flip_u = bool(trials[1] < 0.6 * trials[0])
        if self.flip_u:
            print("  NOTE: left/right looked mirrored - corrected automatically.")
        self.rebuild()

    def targets(self):
        """Clicks corrected for the fact that you click the VISIBLE part of a tooth
        (back teeth are partly hidden), so model reference points can be fitted."""
        return {f: (x - self.bias.get(f, 0j).real, y - self.bias.get(f, 0j).imag)
                for f, (x, y) in self.clicks.items()}

    def _visible_bias(self):
        H, W = self.photo.shape[:2]
        pose = dict(self.pose)
        pose.update(g_scale=1.0, g_roll=0.0, g_dx=0.0, g_dy=0.0, center=self.center,
                    offs=None, scales=None)
        rend = self.renderer.render(pose, W, H, 1.0)
        ys, xs = np.nonzero(rend["lab"])
        ids = rend["lab"][ys, xs].astype(np.int64)
        n = len(self.renderer.fdis) + 1
        cnt = np.bincount(ids, minlength=n).astype(float)
        cx = np.bincount(ids, weights=xs, minlength=n) / np.maximum(cnt, 1)
        cy = np.bincount(ids, weights=ys, minlength=n) / np.maximum(cnt, 1)
        bias = {}
        for f in self.cam:
            i = self.renderer.label_id[f]
            if cnt[i] >= 20:
                rx, ry = rend["ref_xy"][f]
                bias[f] = complex(cx[i] - rx, cy[i] - ry)
        return bias

    def rebuild(self):
        self.cam = to_camera_frame(self.teeth, self.frame, self.flip_u, self.flip_v)
        self.renderer = ArchRenderer(self.cam)
        self.center = self._center()
        self.bias = {}
        self.pose = fit_pose(self.cam, self.clicks)
        for _ in range(2):                      # calibrate: fit -> measure -> refit
            self.bias = self._visible_bias()
            self.pose = fit_pose(self.cam, self.targets())
        self.bias = self._visible_bias()
        self.scales = None
        if self.contours:
            self.compute_tooth_scales()          # available on the Z key; off by default

    def compute_tooth_scales(self):
        """Per-tooth size so each 3D tooth matches its annotated 2D outline."""
        self.scales_on = False
        pose = self.pose_for()
        H, W = self.photo.shape[:2]
        lab = self.renderer.render(pose, W, H, 1.0)["lab"]
        scales = {}
        for f, cnt in self.contours.items():
            if f not in self.cam:
                continue
            cnt = np.asarray(cnt, dtype=float)
            if len(cnt) < 3:
                continue
            wc, hc = np.ptp(cnt[:, 0]), np.ptp(cnt[:, 1])
            ys, xs = np.nonzero(lab == self.renderer.label_id[f])
            if len(xs) < 20 or wc < 2 or hc < 2:
                continue
            wr, hr = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            scales[f] = (float(np.clip(wc / wr, 0.6, 1.6)), float(np.clip(hc / hr, 0.6, 1.6)))
        self.scales = scales

    def pose_for(self, g_scale=1.0, g_roll=0.0, g_dx=0.0, g_dy=0.0,
                 pv=None, pitch=None, yaw=None, tw=1.0, th=1.0, unfold=0.0):
        p = dict(self.pose)
        if pv is not None:
            p["q"], p["persp"] = pv / 1e4, pv
        if pitch is not None:
            p["pitch"] = pitch
        if yaw is not None:
            p["yaw"] = yaw
        p.update(g_scale=g_scale, g_roll=g_roll, g_dx=g_dx, g_dy=g_dy, center=self.center,
                 tw=tw, th=th, unfold=unfold)
        p["offs"] = (snap_offsets(self.cam, self.targets(), self.pose,
                                  p["persp"], p["pitch"], p["yaw"])
                     if self.snap else None)
        p["scales"] = self.scales if (self.scales_on and self.scales) else None
        return p

    def render_full(self, pose, opacity):
        H, W = self.photo.shape[:2]
        rend = self.renderer.render(pose, W, H, 1.0)
        return rend, compose(self.photo, rend, opacity, self.outline, self.labels)


# --------------------------------------------------------------------------
# Guided clicking
# --------------------------------------------------------------------------
class Guide:
    """Holds the camera-frame teeth so the click window can predict positions."""

    def __init__(self, teeth, frame):
        self.teeth, self.frame = teeth, frame
        self.flip_u = False
        self.flip_v = bool(frame.get("auto_flip_v", False))
        self.rebuild()

    def rebuild(self):
        self.cam = to_camera_frame(self.teeth, self.frame, self.flip_u, self.flip_v)
        self.renderer = ArchRenderer(self.cam)

    def u(self, f):
        return float(self.cam[f]["ref"][0])


def click_order(fdis):
    """Centre-out, alternating sides: 11,21,12,22,13,23 ... (or 41,31,42,...)."""
    upper = sum(1 for f in fdis if f // 10 in (1, 2))
    lower = sum(1 for f in fdis if f // 10 in (3, 4))
    qa, qb = (1, 2) if upper >= lower else (4, 3)
    a = sorted(f for f in fdis if f // 10 == qa and f % 10 <= MAX_CLICK_INDEX)
    b = sorted(f for f in fdis if f // 10 == qb and f % 10 <= MAX_CLICK_INDEX)
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def load_clicks_file(path):
    """Clicks saved by an earlier run (final_pose.json) -> {fdi: (x, y)}"""
    with open(path) as fh:
        data = json.load(fh)
    raw = data["clicks"] if "clicks" in data else data
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def load_photo(path):
    img = cv2.imread(path)
    if img is None or img.size == 0:
        raise FileNotFoundError(f"Could not read the photo: {path} "
                                f"(missing, empty or corrupted file)")
    return img


def default_pose_dict(a0, b0, q=DEFAULT_PERSP / 1e4):
    return dict(a0=a0, b0=b0, q=q, persp=q * 1e4, pitch=0, yaw=0, g_scale=1.0,
                g_roll=0.0, g_dx=0.0, g_dy=0.0, center=0j, offs=None, scales=None)


def make_inset(guide, target, iw, ih):
    """Small front-view picture of the 3D arch with the target tooth in red."""
    ren = guide.renderer
    fid = np.unique(ren.F)
    z, _ = plane_coords(ren.V[fid], DEFAULT_PERSP / 1e4, 0, 0)
    x0, x1, y0, y1 = z.real.min(), z.real.max(), z.imag.min(), z.imag.max()
    pad = 10
    s = min((iw - 2 * pad) / max(x1 - x0, 1e-6), (ih - 2 * pad) / max(y1 - y0, 1e-6))
    b0 = complex(iw / 2 - s * (x0 + x1) / 2, ih / 2 - s * (y0 + y1) / 2)
    rend = ren.render(default_pose_dict(complex(s, 0), b0), iw, ih, 1.0)
    lab, shade = rend["lab"], rend["shade"]
    img = np.full((ih, iw, 3), 45, np.uint8)
    m = lab > 0
    img[m] = np.clip(shade[m].astype(np.int32) * 0.75 + 40, 0, 255).astype(np.uint8)
    tm = lab == ren.label_id[target]
    img[tm] = (0.3 * img[tm] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
    img[outline_mask(lab)] = (255, 255, 255)
    for f, (x, y) in rend["ref_xy"].items():
        if f % 10 <= MAX_CLICK_INDEX:
            col = (0, 255, 255) if f == target else (255, 255, 255)
            cv2.putText(img, str(f), (int(x) - 9, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(f), (int(x) - 9, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, col, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (iw - 1, ih - 1), (200, 200, 200), 1)
    return img


def predict_positions(guide, clicks):
    """Where the model expects each tooth (full-res px), from the clicks so far."""
    if len(clicks) < 2:
        return None, None
    pose = fit_pose(guide.cam, clicks, extended=False, min_points=2)
    out = {}
    for f in guide.cam:
        z, _ = plane_coords(guide.cam[f]["ref"][None, :], pose["q"], 0, 0)
        p = pose["a0"] * z[0] + pose["b0"]
        out[f] = (p.real, p.imag)
    return out, pose


def check_click(guide, clicks, f, x, y):
    """Return (ok, message, forceable)."""
    for g, (gx, gy) in clicks.items():
        du = guide.u(f) - guide.u(g)
        if abs(du) > 0.5 and (x - gx) * du < 0:
            side = "RIGHT" if du > 0 else "LEFT"
            return (False, f"Not recorded: tooth {f} must be to the {side} of tooth {g}.", False)
    if len(clicks) >= 3:
        pred, pose = predict_positions(guide, clicks)
        s = abs(pose["a0"])
        d = math.hypot(x - pred[f][0], y - pred[f][1]) / max(s, 1e-9)
        if d > 10.0:
            return (False, f"Far from where tooth {f} should be (~{d:.0f} mm off). "
                           f"Wrong row/tooth? Press F to keep it anyway.", True)
    return True, "", False


def collect_clicks(photo, guide, allow_skip=False):
    H, W = photo.shape[:2]
    ps = min(1.0, MAX_DISPLAY / W, MAX_DISPLAY / H)
    sw, sh = int(W * ps), int(H * ps)
    small = cv2.resize(photo, (sw, sh), interpolation=cv2.INTER_AREA)
    order = click_order(list(guide.teeth))
    if len(order) < 4:
        raise ValueError("Not enough teeth in the 3D model to click")
    iw = min(440, int(sw * 0.62))
    ih = int(iw * 0.30)
    st = {"i": 0, "hist": [], "clicks": {}, "msg": "", "pending": None,
          "base": None, "pred": None, "dirty": True, "corner": 0}
    inset_cache = {}
    win = "Click the teeth"

    def target():
        return order[st["i"]] if st["i"] < len(order) else None

    def record(f, x, y):
        st["clicks"][f] = (x, y)
        st["hist"].append(("c", f))
        st["i"] += 1
        st["msg"], st["pending"], st["dirty"] = "", None, True

    def on_mouse(ev, x, y, flags, param):
        f = target()
        if ev != cv2.EVENT_LBUTTONDOWN or f is None:
            return
        fx, fy = x / ps, y / ps
        ok, msg, forceable = check_click(guide, st["clicks"], f, fx, fy)
        if ok:
            record(f, fx, fy)
        else:
            st["msg"] = msg
            st["pending"] = (f, fx, fy) if forceable else None

    def rebuild_base():
        base, pred = small.copy(), None
        pred_full, pose = predict_positions(guide, st["clicks"])
        if pred_full is not None:
            pp = default_pose_dict(pose["a0"], pose["b0"], pose["q"])
            rend = guide.renderer.render(pp, sw, sh, ps)
            base = compose(small, rend, 0.0, True, False)
            pred = {f: (x * ps, y * ps) for f, (x, y) in pred_full.items()}
        st["base"], st["pred"], st["dirty"] = base, pred, False

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    print("\nClick the CENTRE of each tooth's visible crown. The small picture shows")
    print("which tooth (red). Yellow marks show where the model EXPECTS the teeth.")
    print("  a = accept expected spot   n = skip   u = undo   f = keep far click")
    print("  h = mirror   v = flip up/down   m = move the small picture   Enter = finish (>= 4 teeth)\n")
    while True:
        if st["dirty"]:
            rebuild_base()
        fr = st["base"].copy()
        f = target()
        for g, (x, y) in st["clicks"].items():
            p = (int(x * ps), int(y * ps))
            cv2.circle(fr, p, 5, (0, 0, 255), -1)
            cv2.putText(fr, str(g), (p[0] + 7, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2, cv2.LINE_AA)
        if st["pred"] is not None:
            for g, (x, y) in st["pred"].items():
                if g not in st["clicks"] and g % 10 <= MAX_CLICK_INDEX:
                    cv2.circle(fr, (int(x), int(y)), 3, (0, 255, 255), -1)
                    cv2.putText(fr, str(g), (int(x) + 5, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 255), 1, cv2.LINE_AA)
            if f is not None and f in st["pred"]:
                cv2.circle(fr, (int(st["pred"][f][0]), int(st["pred"][f][1])),
                           16, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(fr, (0, 0), (sw, 34), (0, 0, 0), -1)
        if f is not None:
            side = "LEFT" if f // 10 in (1, 4) else "RIGHT"
            msg = (f"Click tooth {f} ({st['i'] + 1}/{len(order)}): {QUADRANT_NAMES[f // 10]} "
                   f"{TOOTH_NAMES[f % 10]} - {side} of the photo's centre")
            if f % 10 == 1:
                msg = (f"Click tooth {f} ({st['i'] + 1}/{len(order)}): the big front tooth "
                       f"just {side} of the centre line")
        else:
            msg = "All done - press Enter"
        cv2.putText(fr, msg, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 255, 255), 1, cv2.LINE_AA)
        if f is not None:
            if f not in inset_cache:
                inset_cache[f] = make_inset(guide, f, iw, ih)
            cx0 = 8 if st["corner"] in (0, 2) else max(8, sw - iw - 8)
            cy0 = 40 if st["corner"] in (0, 1) else max(40, sh - ih - 40)
            fr[cy0:cy0 + ih, cx0:cx0 + iw] = inset_cache[f]
        if allow_skip and not st["msg"]:
            cv2.putText(fr, "k = skip this jaw", (8, sh - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if st["msg"]:
            cv2.rectangle(fr, (0, sh - 30), (sw, sh), (0, 0, 0), -1)
            cv2.putText(fr, st["msg"], (8, sh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (80, 80, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, fr)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, ord("q")):
            if len(st["clicks"]) >= 4:
                break
            st["msg"] = f"Need at least 4 teeth (have {len(st['clicks'])})."
        elif k == ord("k") and allow_skip:
            st["clicks"] = {}
            break
        elif k == ord("n") and f is not None:
            st["hist"].append(("n", f))
            st["i"] += 1
        elif k == ord("u") and st["hist"]:
            kind, g = st["hist"].pop()
            st["i"] -= 1
            if kind == "c":
                st["clicks"].pop(g, None)
            st["msg"], st["pending"], st["dirty"] = "", None, True
        elif k == ord("m"):
            st["corner"] = (st["corner"] + 1) % 4
        elif k == ord("a") and f is not None and st["pred"] is not None:
            x, y = st["pred"][f]
            record(f, x / ps, y / ps)
        elif k == ord("f") and st["pending"] is not None:
            g, x, y = st["pending"]
            record(g, x, y)
        elif k in (ord("h"), ord("v")):
            if k == ord("h"):
                guide.flip_u = not guide.flip_u
            else:
                guide.flip_v = not guide.flip_v
            guide.rebuild()
            inset_cache.clear()
            st.update(i=0, hist=[], clicks={}, msg="Orientation changed - clicks cleared.",
                      pending=None, dirty=True)
    cv2.destroyWindow(win)
    return dict(st["clicks"])


# --------------------------------------------------------------------------
# Dataset mode (annotated photos: no clicking needed)
# --------------------------------------------------------------------------
def resolve_patient(arg):
    if os.path.isdir(arg):
        return arg
    key = arg if not arg.isdigit() else f"Patient_{arg}"
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(os.getcwd(), "patient_index.json"),
              os.path.join(here, "patient_index.json")):
        if os.path.exists(p):
            with open(p) as fh:
                idx = json.load(fh)
            if key in idx:
                return idx[key]
    raise FileNotFoundError(
        f"'{arg}' is not a folder and was not found in patient_index.json")


def load_patient(arg):
    """-> photo_path, {fdi: (x, y)} centroids, {fdi: contour Nx2}"""
    folder = resolve_patient(arg)
    jp = os.path.join(folder, "tooth_contours.json")
    pp = os.path.join(folder, "original.png")
    if not os.path.exists(jp) or not os.path.exists(pp):
        raise FileNotFoundError(f"Need original.png and tooth_contours.json in {folder}")
    with open(jp) as fh:
        data = json.load(fh)
    cents, conts = {}, {}
    for t in data["teeth"]:
        f = int(t["fdi"])
        cents[f] = (float(t["centroid"]["x"]), float(t["centroid"]["y"]))
        conts[f] = np.asarray(t["contour"], dtype=float)
    return pp, cents, conts


# --------------------------------------------------------------------------
# Results + fine-tune window
# --------------------------------------------------------------------------
def jaw_name(sess):
    return "upper" if sess.frame.get("upper", True) else "lower"


def layer_rgba(rend, opacity):
    mask = rend["lab"] > 0
    rgba = np.dstack([rend["shade"], (mask * int(255 * opacity)).astype(np.uint8)])
    rgba[outline_mask(rend["lab"])] = (*OUTLINE_COLOR, 255)
    return rgba


def pose_info(sess, pose):
    return {
        "case_dir": sess.case_dir,
        "jaw": jaw_name(sess),
        "clicks": {str(k): list(v) for k, v in sess.clicks.items()},
        "tooth_scales": {str(k): list(v) for k, v in (sess.scales or {}).items()},
        "flip_u": sess.flip_u, "flip_v": sess.flip_v, "snap": sess.snap,
        "pose": {"a0": [pose["a0"].real, pose["a0"].imag],
                 "b0": [pose["b0"].real, pose["b0"].imag],
                 "perspective": pose["persp"], "pitch": pose["pitch"], "yaw": pose["yaw"],
                 "g_scale": pose["g_scale"], "g_roll": pose["g_roll"],
                 "g_dx": pose["g_dx"], "g_dy": pose["g_dy"],
                 "tooth_width": pose.get("tw", 1.0), "tooth_height": pose.get("th", 1.0),
                 "unfold": pose.get("unfold", 0.0)},
    }


def save_results(sess, pose, opacity, tag="final"):
    rend, out = sess.render_full(pose, opacity)
    cv2.imwrite(f"{tag}_overlay.png", out)
    cv2.imwrite(f"{tag}_overlay_rgba.png", layer_rgba(rend, opacity))
    info = pose_info(sess, pose)
    info["photo_size"] = [int(sess.photo.shape[1]), int(sess.photo.shape[0])]
    info["opacity"] = opacity
    with open(f"{tag}_pose.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved {tag}_overlay.png, {tag}_overlay_rgba.png, {tag}_pose.json")


def draw_order(items):
    """Lower jaw first, upper jaw on top (the upper teeth overlap the lower ones)."""
    return sorted(range(len(items)), key=lambda i: 0 if jaw_name(items[i][0]) == "lower" else 1)


def save_results_multi(items, poses, opacity, tag="final"):
    photo = items[0][0].photo
    H, W = photo.shape[:2]
    out, rgba = photo.copy(), np.zeros((H, W, 4), np.uint8)
    jaws, clicks = {}, {}
    for i in draw_order(items):
        s = items[i][0]
        rend = s.renderer.render(poses[i], W, H, 1.0)
        out = compose(out, rend, opacity, s.outline, s.labels)
        layer = layer_rgba(rend, opacity)
        m = layer[:, :, 3] > 0
        rgba[m] = layer[m]
        info = pose_info(s, poses[i])
        clicks.update(info["clicks"])
        jaws[jaw_name(s)] = info
    cv2.imwrite(f"{tag}_overlay.png", out)
    cv2.imwrite(f"{tag}_overlay_rgba.png", rgba)
    with open(f"{tag}_pose.json", "w") as f:
        json.dump({"photo_size": [int(W), int(H)], "clicks": clicks, "jaws": jaws,
                   "opacity": opacity}, f, indent=2)
    print(f"Saved {tag}_overlay.png, {tag}_overlay_rgba.png, {tag}_pose.json "
          f"({' + '.join(jaws)})")


def run_gui(sess, guide=None, extra=None):
    """Fine-tune window.  `extra` = more (session, guide) pairs, e.g. the lower jaw."""
    items = [(sess, guide)] + list(extra or [])
    multi = len(items) > 1
    H, W = sess.photo.shape[:2]
    ps = min(1.0, MAX_DISPLAY / W, MAX_DISPLAY / H)
    sw, sh = int(W * ps), int(H * ps)
    small = cv2.resize(sess.photo, (sw, sh), interpolation=cv2.INTER_AREA)
    win, ctrl = "Fine-tune (photo)", "Controls"
    noop = lambda v: None
    bars = [("Scale %", 100, 200), ("Move X", 200, 400), ("Move Y", 200, 400),
            ("Roll", 30, 60), ("Perspective", DEFAULT_PERSP, 100),
            ("Pitch", 30, 60), ("Yaw", 30, 60), ("Opacity %", 55, 100),
            ("Tooth width %", 100, 200), ("Tooth height %", 100, 200),
            ("EXPERIMENTAL turn sides %", 0, 100)]
    ui = {"drag": None, "rends": [None] * len(items), "ver": 0, "active": 0}
    stored = [None] * len(items)           # slider values of a jaw while it is not the active one
    order = draw_order(items)
    help_lines = ["KEYS", "t  snap teeth to clicks", "z  per-tooth size (dataset)",
                  "o  outlines    l  labels", "c  compare with plain photo",
                  "h  mirror      v  flip up/down", "x  redo clicks   r  reset",
                  "s  SAVE        q  quit", "MOUSE: drag a tooth onto the real one"]
    if multi:
        help_lines.insert(1, "j  switch jaw: sliders move the ACTIVE jaw")

    def active_sess():
        return items[ui["active"]][0]

    def default_vals(s):
        return {"Scale %": 100, "Move X": 200, "Move Y": 200, "Roll": 30,
                "Perspective": int(s.pose["persp"]),
                "Pitch": 30 + int(s.pose["pitch"]), "Yaw": 30 + int(s.pose["yaw"]),
                "Opacity %": 55, "Tooth width %": 100, "Tooth height %": 100,
                "EXPERIMENTAL turn sides %": 0}

    def set_bars(vals):
        for n, _, mx in bars:
            cv2.setTrackbarPos(n, ctrl, int(np.clip(vals[n], 0, mx)))

    def read_bars():
        return {n: cv2.getTrackbarPos(n, ctrl) for n, _, _ in bars}

    def reset():
        set_bars(default_vals(active_sess()))

    def pose_from(s, v):
        return s.pose_for(max(0.1, v["Scale %"] / 100.0), v["Roll"] - 30,
                          (v["Move X"] - 200) / ps, (v["Move Y"] - 200) / ps,
                          v["Perspective"], v["Pitch"] - 30, v["Yaw"] - 30,
                          max(0.3, v["Tooth width %"] / 100.0),
                          max(0.3, v["Tooth height %"] / 100.0),
                          v["EXPERIMENTAL turn sides %"] / 100.0)

    def on_mouse(ev, x, y, flags, param):
        """Drag a tooth (of either jaw) onto the real one; needs snap ON."""
        if not active_sess().snap:
            return
        if ev == cv2.EVENT_LBUTTONDOWN:
            best, bd = None, 45.0
            for i, rend in enumerate(ui["rends"]):
                if rend is None:
                    continue
                for f, (rx, ry) in rend["ref_xy"].items():
                    d = math.hypot(rx - x, ry - y)
                    if d < bd:
                        best, bd = (i, f), d
            ui["drag"] = None
            if best is not None:
                i, f = best
                s = items[i][0]
                if f in s.clicks:
                    base = s.clicks[f]
                else:
                    rx, ry = ui["rends"][i]["ref_xy"][f]
                    b = s.bias.get(f, 0j)
                    base = (rx / ps + b.real, ry / ps + b.imag)
                ui["drag"] = (i, f, x, y, base)
        elif ev == cv2.EVENT_MOUSEMOVE and ui["drag"] is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            i, f, x0, y0, base = ui["drag"]
            items[i][0].clicks[f] = (base[0] + (x - x0) / ps, base[1] + (y - y0) / ps)
            ui["ver"] += 1
        elif ev == cv2.EVENT_LBUTTONUP:
            ui["drag"] = None

    def ctrl_image():
        img = np.full((20 + 18 * len(help_lines) + 8 + 20 * len(items), 430, 3), 40, np.uint8)
        for i, line in enumerate(help_lines):
            cv2.putText(img, line, (10, 22 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (0, 255, 255) if i == 0 else (235, 235, 235), 1, cv2.LINE_AA)
        for k, (s, _) in enumerate(items):
            q = s.pose["rms_mm"]
            good = q < 1.5
            y = 22 + 18 * len(help_lines) + 8 + 20 * k
            cv2.putText(img, f"{jaw_name(s)} fit {q:.1f} mm  ({'good' if good else 'poor: press x'})",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 0) if good else (60, 60, 255), 1, cv2.LINE_AA)
        return img

    def make_windows():
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(ctrl, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(win, 10, 10)
        cv2.moveWindow(ctrl, sw + 30, 10)
        cv2.imshow(ctrl, ctrl_image())
        for n, v0, mx in bars:
            cv2.createTrackbar(n, ctrl, v0, mx, noop)
        cv2.setMouseCallback(win, on_mouse)
        reset()

    make_windows()
    print("\nTwo windows: the photo, and 'Controls' (sliders + key list) to its right.")
    print("Drag a green tooth onto the real tooth (needs snap ON). s = save, q = quit.")
    if multi:
        print("Two jaws loaded: press j to switch which jaw the sliders move.")
    for s, _ in items:
        q = s.pose["rms_mm"]
        print(f"Fit {jaw_name(s)}: {s.pose['n_used']} teeth, residual {q:.2f} mm "
              f"(perspective {s.pose['persp']}, pitch {s.pose['pitch']}, yaw {s.pose['yaw']})")
        if q > 4.0:
            print(f"  WARNING ({jaw_name(s)}): the clicks don't match a normal tooth arrangement "
                  f"well. Press x to redo them (check which tooth each prompt asked for).")

    def all_vals(v):
        return [v if i == ui["active"] else (stored[i] or default_vals(items[i][0]))
                for i in range(len(items))]

    last, out = None, small
    while True:
        v = read_bars()
        vals = all_vals(v)
        key = (tuple(tuple(x.values()) for x in vals),
               tuple((s.snap, s.outline, s.labels, s.compare, s.flip_u, s.flip_v, s.scales_on)
                     for s, _ in items), ui["ver"], ui["active"])
        if key != last:
            opacity = v["Opacity %"] / 100.0
            cur = small.copy()
            if not active_sess().compare:
                for i in order:
                    s = items[i][0]
                    rend = s.renderer.render(pose_from(s, vals[i]), sw, sh, ps)
                    ui["rends"][i] = rend
                    cur = compose(cur, rend, opacity, s.outline, s.labels)
            a = active_sess()
            tag = (f"{('jaw: ' + jaw_name(a) + '  ') if multi else ''}"
                   f"snap:{'ON' if a.snap else 'off'}  size:{'ON' if a.scales_on else 'off'}"
                   f"  outlines:{'ON' if a.outline else 'off'}"
                   f"{'  [showing plain photo]' if a.compare else ''}")
            cv2.putText(cur, tag, (8, sh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(cur, tag, (8, sh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            out = cur
            last = key
        cv2.imshow(win, out)
        k = cv2.waitKey(25) & 0xFF
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        a = active_sess()
        if k in (ord("q"), 27):
            break
        elif k == ord("j") and multi:
            stored[ui["active"]] = dict(v)
            ui["active"] = (ui["active"] + 1) % len(items)
            nv = dict(stored[ui["active"]] or default_vals(active_sess()))
            nv["Opacity %"] = v["Opacity %"]           # opacity is shared by both jaws
            set_bars(nv)
            ui["ver"] += 1
        elif k in (ord("t"), ord("o"), ord("l"), ord("c")):
            attr = {ord("t"): "snap", ord("o"): "outline", ord("l"): "labels",
                    ord("c"): "compare"}[k]
            new = not getattr(a, attr)
            for s, _ in items:
                setattr(s, attr, new)
        elif k == ord("z") and any(s.scales for s, _ in items):
            new = not a.scales_on
            for s, _ in items:
                if s.scales:
                    s.scales_on = new
        elif k in (ord("h"), ord("v")):
            if k == ord("h"):
                a.flip_u = not a.flip_u
            else:
                a.flip_v = not a.flip_v
            a.rebuild()
            reset()
            last = None
        elif k == ord("x") and items[ui["active"]][1] is not None:
            g = items[ui["active"]][1]
            cv2.destroyWindow(win)
            cv2.destroyWindow(ctrl)
            new = collect_clicks(a.photo, g)
            a.clicks = dict(new)
            a.flip_u, a.flip_v = g.flip_u, g.flip_v
            a.rebuild()
            make_windows()
            last = None
        elif k == ord("r"):
            reset()
        elif k == ord("s"):
            v = read_bars()
            vals = all_vals(v)
            poses = [pose_from(items[i][0], vals[i]) for i in range(len(items))]
            if multi:
                save_results_multi(items, poses, v["Opacity %"] / 100.0)
            else:
                save_results(sess, poses[0], v["Opacity %"] / 100.0)
    cv2.destroyAllWindows()


def derive_lower(case):
    """Find the lower-arch folder that sits next to the upper one (same case ID)."""
    for a, b in (("processed_part1_upper", "processed_part1_lower"), ("upper", "lower")):
        if a in case:
            cand = case.replace(a, b)
            if os.path.isdir(cand):
                return cand
    return None


def find_lower(case):
    """Lower-arch folder for --both.  Same case ID if it exists, otherwise any usable lower
    case in the sibling library (the photo is a different person anyway, so any arch can serve)."""
    same = derive_lower(case)
    if same:
        return same, "same case ID"
    for a, b in (("processed_part1_upper", "processed_part1_lower"), ("upper", "lower")):
        if a in case:
            lib = os.path.dirname(case.replace(a, b))
            if os.path.isdir(lib):
                for name in sorted(os.listdir(lib)):
                    p = os.path.join(lib, name)
                    if os.path.isdir(p) and len(glob.glob(os.path.join(p, "tooth_*.obj"))) >= 8:
                        return p, "picked automatically (no lower folder with the upper case's ID)"
    return None, ""


def main():
    ap = argparse.ArgumentParser(description="Overlay 3D tooth arches on a frontal photo")
    ap.add_argument("--case", default=DEFAULT_CASE, help="folder with tooth_*.obj (upper arch)")
    ap.add_argument("--case-lower", default=None, help="folder with the LOWER arch tooth_*.obj")
    ap.add_argument("--both", action="store_true",
                    help="also load the matching lower arch automatically (same case ID)")
    ap.add_argument("--photo", default=DEFAULT_PHOTO, help="your own frontal photo (click mode)")
    ap.add_argument("--clicks", default=None,
                    help="re-use clicks saved by an earlier run (final_pose.json): no clicking")
    ap.add_argument("--patient", default=None,
                    help="dataset patient: a folder (original.png + tooth_contours.json) "
                         "or an ID such as Patient_1 / 1  -> NO clicking needed")
    args = ap.parse_args()

    print(f"dental_overlay {VERSION}")
    case_dirs = [args.case]
    lower, how_lower = (args.case_lower, "given with --case-lower") if args.case_lower else \
        (find_lower(args.case) if args.both else (None, ""))
    if args.both and not lower:
        raise SystemExit("--both: no lower-arch folder found next to the upper one.\n"
                         "Run  python check_lower.py  to see what is on disk, or pass the folder "
                         "yourself with  --case-lower \"path\\to\\processed_part1_lower\\SOMECASE\"")
    if lower:
        case_dirs.append(lower)
        print(f"Upper arch: {args.case}")
        print(f"Lower arch: {lower}   [{how_lower}]")
    loaded = []
    for cd in case_dirs:
        print(f"Loading 3D teeth from {os.path.basename(cd)} (a few seconds) ...")
        t = prepare_teeth(load_case(cd))
        loaded.append((cd, t, anatomical_frame(t)))
    if len(loaded) == 2 and loaded[0][2]["upper"] == loaded[1][2]["upper"]:
        raise SystemExit("Both folders are the same jaw - give one upper and one lower arch.")

    items = []
    if args.patient:
        photo_path, cents, conts = load_patient(args.patient)
        photo = load_photo(photo_path)
        print(f"Dataset photo {photo.shape[1]}x{photo.shape[0]}")
        for cd, teeth, frame in loaded:
            clicks = {f: p for f, p in cents.items() if f in teeth and f % 10 <= MAX_CLICK_INDEX}
            if len(clicks) < 4:
                print(f"  {os.path.basename(cd)}: only {len(clicks)} annotated teeth match - skipped "
                      f"(this patient may not show that jaw).")
                continue
            contours = {f: c for f, c in conts.items() if f in teeth}
            s = Session(teeth, frame, photo, clicks, cd, contours=contours)
            g = Guide(teeth, frame)
            g.flip_u, g.flip_v = s.flip_u, s.flip_v
            g.rebuild()
            items.append((s, g))
            print(f"  {jaw_name(s)}: {len(clicks)} annotated teeth in use")
    elif args.clicks:
        photo = load_photo(args.photo)
        saved = load_clicks_file(args.clicks)
        print(f"Re-using saved clicks on {photo.shape[1]}x{photo.shape[0]} photo")
        for cd, teeth, frame in loaded:
            clicks = {f: p for f, p in saved.items() if f in teeth}
            if len(clicks) < 4:
                print(f"  {os.path.basename(cd)}: fewer than 4 saved clicks match - skipped")
                continue
            s = Session(teeth, frame, photo, clicks, cd)
            g = Guide(teeth, frame)
            g.flip_u, g.flip_v = s.flip_u, s.flip_v
            g.rebuild()
            items.append((s, g))
    else:
        photo = load_photo(args.photo)
        print(f"Photo: {photo.shape[1]}x{photo.shape[0]} px")
        for idx, (cd, teeth, frame) in enumerate(loaded):
            g = Guide(teeth, frame)
            if idx > 0:
                print("\nNow the other jaw. Press k to skip it.")
            clicks = collect_clicks(photo, g, allow_skip=(idx > 0))
            clicks = {f: p for f, p in clicks.items() if f in teeth}
            if len(clicks) < 4:
                if idx == 0:
                    raise SystemExit("Fewer than 4 teeth clicked - run again.")
                print("  Lower jaw skipped.")
                continue
            items.append((Session(teeth, frame, photo, clicks, cd,
                                  flip_u=g.flip_u, flip_v=g.flip_v), g))
    if not items:
        raise SystemExit("Nothing to show - no jaw had enough matching teeth.")
    run_gui(items[0][0], items[0][1], extra=items[1:])


if __name__ == "__main__":
    main()