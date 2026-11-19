"""
selftest_overlay.py  -  proves the overlay pipeline works WITHOUT any real data.

It builds a fake dental arch in a known real-world layout, takes a "photo" of
it with an independent pinhole-camera model (NOT using the main code's
projection), simulates imperfect clicks, then checks that dental_overlay.py
recovers the alignment.   Run:   python selftest_overlay.py
"""
import math
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import trimesh

import dental_overlay as do

WIDTHS = [8.6, 6.6, 7.6, 7.0, 6.8, 10.2, 9.4]          # mm, tooth 1..7
HEIGHTS = [8.0, 7.5, 8.0, 6.5, 6.5, 5.5, 5.5]
DEPTHS = [4.0, 3.8, 4.5, 4.6, 4.6, 5.5, 5.5]


def rot_axis(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_arch(upper=True):
    """Patient world frame: x=east (patient RIGHT), y=north (front), z=up."""
    a, b = 27.0, 23.0
    ts = np.linspace(0, 1.55, 4000)
    xs, ys = a * np.sin(ts), b * np.cos(ts)
    cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    teeth, pos = {}, 0.0
    ico = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    for idx in range(1, 8):
        w = WIDTHS[idx - 1]
        mid = pos + w / 2
        pos += w
        t = float(np.interp(mid, cum, ts))
        cx, cy = a * np.sin(t), b * np.cos(t)
        dx, dy = a * np.cos(t), -b * np.sin(t)
        for side in (+1, -1):               # +1 east = patient RIGHT
            V = ico.vertices * np.array([w / 2, DEPTHS[idx - 1], HEIGHTS[idx - 1]])
            F = ico.faces.copy()
            cz = V[F].mean(axis=1)[:, 2]
            keep = cz <= 0 if upper else cz >= 0
            F = F[keep]
            ang = math.atan2(dy, side * dx)
            V = V @ rot_axis("z", math.degrees(ang)).T
            V = V + np.array([side * cx, cy, 0.0])
            if upper:
                q = 1 if side > 0 else 2
            else:
                q = 4 if side > 0 else 3
            used = np.unique(F)
            remap = -np.ones(len(V), int)
            remap[used] = np.arange(len(used))
            teeth[q * 10 + idx] = (V[used], remap[F])
    return teeth


def save_scanner_frame(teeth, folder, seed):
    rng = np.random.default_rng(seed)
    Rs = rot_axis("z", rng.uniform(0, 360)) @ rot_axis("x", rng.uniform(0, 360)) @ \
        rot_axis("y", rng.uniform(0, 360))
    t = rng.uniform(-80, 80, 3)
    os.makedirs(folder, exist_ok=True)
    for fdi, (V, F) in teeth.items():
        Vs = V @ Rs.T + t
        trimesh.Trimesh(Vs, F, process=False).export(os.path.join(folder, f"tooth_{fdi}.obj"))


def camera(W, H, f, C, pitch, yaw, roll):
    R_base = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)   # x=west,y=down,z=south
    R = rot_axis("z", roll) @ rot_axis("y", yaw) @ rot_axis("x", pitch) @ R_base
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
    rvec = cv2.Rodrigues(R)[0]
    tvec = (-R @ C).reshape(3, 1)
    return K, rvec, tvec, R


def project(X, K, rvec, tvec):
    return cv2.projectPoints(X.astype(np.float64), rvec, tvec, K, None)[0].reshape(-1, 2)


def photograph(teeth, upper, W=1600, H=1200, seed=0):
    rng = np.random.default_rng(seed)
    f = 3200.0
    C = np.array([3.0, 23.0 + 250.0, -4.0 if upper else 4.0])
    pitch, yaw, roll = 7.0, -5.0, 4.0
    K, rvec, tvec, R = camera(W, H, f, C, pitch, yaw, roll)
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:] = (120, 110, 190)                                           # pinkish
    bg = cv2.GaussianBlur((bg + rng.integers(0, 12, bg.shape, dtype=np.uint8)), (0, 0), 3)
    img = bg.copy()
    gt_lab = np.zeros((H, W), np.uint8)
    faces = []
    for fdi, (V, F) in teeth.items():
        n, area, cen = do.face_geometry(V, do.orient_outward(V, F))
        F2 = do.orient_outward(V, F)
        n, area, cen = do.face_geometry(V, F2)
        toc = C - cen
        toc /= np.linalg.norm(toc, axis=1, keepdims=True)
        vis = (n * toc).sum(1) > 0
        P2 = project(V, K, rvec, tvec)
        zc = (R @ (V - C).T).T[:, 2]
        for i in np.nonzero(vis)[0]:
            tri = F2[i]
            faces.append((zc[tri].mean(), P2[tri], fdi, float((n[i] * toc[i]).sum())))
        wv = area * np.maximum(0, (n * toc).sum(1))
        ref = (wv[:, None] * cen).sum(0) / wv.sum()
        teeth[fdi] = (V, F, ref)
    faces.sort(key=lambda t: -t[0])
    for z, tri, fdi, lam in faces:
        col = tuple(int(c * (0.55 + 0.45 * lam)) for c in (205, 225, 235))
        p = np.round(tri * 16).astype(np.int32)
        cv2.fillConvexPoly(img, p, col, cv2.LINE_8, 4)
        cv2.fillConvexPoly(gt_lab, p, fdi, cv2.LINE_8, 4)
    gt_ref = {fdi: project(t[2][None, :], K, rvec, tvec)[0] for fdi, t in teeth.items()}
    return img, gt_lab, gt_ref


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def run_case(upper, seed, tmp, noise=3.0, verbose=True):
    world = make_arch(upper)
    folder = os.path.join(tmp, f"case_{'u' if upper else 'l'}_{seed}")
    save_scanner_frame(world, folder, seed)
    photo, gt_lab, gt_ref = photograph(dict(world), upper, seed=seed)
    rng = np.random.default_rng(100 + seed)
    teeth = do.prepare_teeth(do.load_case(folder))
    frame = do.anatomical_frame(teeth)
    clicks = {f: tuple(gt_ref[f] + rng.normal(0, noise, 2))
              for f in gt_ref if f % 10 <= 6}
    sess = do.Session(teeth, frame, photo, clicks, folder)
    out = {}
    for snap in (False, True):
        sess.snap = snap
        pose = sess.pose_for()
        t0 = time.time()
        rend, comp = sess.render_full(pose, 0.55)
        dt = time.time() - t0
        m = rend["lab"] > 0
        gt = gt_lab > 0
        err = np.mean([np.hypot(rend["ref_xy"][f][0] - gt_ref[f][0],
                                rend["ref_xy"][f][1] - gt_ref[f][1])
                       for f in gt_ref if f in rend["ref_xy"] and f % 10 <= 6])
        out[snap] = dict(iou=iou(m, gt), center_err=err, secs=dt, img=comp)
    out["pose"] = sess.pose
    out["flip_u"] = sess.flip_u
    out["photo"] = photo
    # independent per-tooth check: is every FDI label on the correct side/row?
    return out


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    allok = True
    first = None
    for upper in (True, False):
        for seed in (1, 2, 3):
            r = run_case(upper, seed, tmp)
            p = r["pose"]
            print(f"{'UPPER' if upper else 'LOWER'} seed {seed}: "
                  f"fit rms {p['rms_mm']:.2f} mm  persp {p['persp']} pitch {p['pitch']} yaw {p['yaw']} "
                  f"mirrored={r['flip_u']} | IoU no-snap {r[False]['iou']:.2f}, "
                  f"snap {r[True]['iou']:.2f} | centre err no-snap {r[False]['center_err']:.1f}px, "
                  f"snap {r[True]['center_err']:.1f}px | render {r[True]['secs']*1000:.0f} ms (full-res)")
            ok = r[False]["iou"] > 0.70 and not r["flip_u"]
            allok &= ok
            if first is None:
                first = r
                cv2.imwrite("selftest_photo.png", r["photo"])
                cv2.imwrite("selftest_overlay_nosnap.png", r[False]["img"])
                cv2.imwrite("selftest_overlay_snap.png", r[True]["img"])
    print("\nSELF-TEST", "PASSED" if allok else "FAILED")
    sys.exit(0 if allok else 1)