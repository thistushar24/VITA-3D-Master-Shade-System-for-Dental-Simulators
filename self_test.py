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


def make_arch(upper=True, a=27.0, b=23.0):
    """Patient world frame: x=east (patient RIGHT), y=north (front), z=up."""
    ts = np.linspace(0, 2.4, 6000)       # long enough for 7 teeth per side
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


def visible_centres(gt_lab):
    """What a person clicks / what the dataset stores: centre of the VISIBLE crown."""
    out = {}
    for f in np.unique(gt_lab):
        if f:
            ys, xs = np.nonzero(gt_lab == f)
            out[int(f)] = np.array([xs.mean(), ys.mean()])
    return out


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
    gv = visible_centres(gt_lab)
    clicks = {f: tuple(gv[f] + rng.normal(0, noise, 2))
              for f in gv if f % 10 <= 6}
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



# ----------------------------------------------------------------------------
# Fake OpenCV GUI so the real click/fine-tune code can be driven without a screen
# ----------------------------------------------------------------------------
class FakeGUI:
    NAMES = ["namedWindow", "setMouseCallback", "imshow", "waitKey", "destroyWindow",
             "destroyAllWindows", "createTrackbar", "getTrackbarPos", "setTrackbarPos",
             "getWindowProperty", "moveWindow"]

    def __init__(self):
        self.cb, self.tb, self.script, self.win, self.last = {}, {}, [], None, None
        self.cb_last = None
        self.orig = {n: getattr(cv2, n) for n in self.NAMES}

    def install(self):
        cv2.namedWindow = lambda *a, **k: None
        cv2.destroyWindow = lambda *a, **k: None
        cv2.destroyAllWindows = lambda *a, **k: None
        cv2.setMouseCallback = lambda win, cb, *a: setattr(self, 'cb_last', cb)
        cv2.moveWindow = lambda *a, **k: None
        cv2.imshow = self._imshow
        cv2.createTrackbar = lambda n, w, v, m, cb: self.tb.__setitem__(n, v)
        cv2.getTrackbarPos = lambda n, w: self.tb[n]
        cv2.setTrackbarPos = lambda n, w, v: self.tb.__setitem__(n, v)
        cv2.getWindowProperty = lambda *a: 1
        cv2.waitKey = self._waitkey

    def restore(self):
        for n, f in self.orig.items():
            setattr(cv2, n, f)

    def _imshow(self, win, img):
        self.win = win
        if 'Controls' not in win:
            self.last = img

    def _waitkey(self, ms):
        if not self.script:
            raise AssertionError("test script exhausted (UI waiting for more input)")
        act = self.script.pop(0)
        if act[0] == "key":
            return act[1]
        if act[0] == "mouse":
            self.cb_last(act[1], int(round(act[2])), int(round(act[3])), act[4], None)
        if act[0] == "call":
            act[1]()
        return 255


def click(x, y):
    return ("mouse", cv2.EVENT_LBUTTONDOWN, x, y, 0)


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")
    return bool(cond)


def make_gt_patient_folder(folder, photo, gt_lab, noise=0.0, rng=None):
    """Write original.png + tooth_contours.json like the IOP Compass dataset."""
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, "original.png"), photo)
    teeth = []
    for fdi in np.unique(gt_lab):
        if fdi == 0:
            continue
        m = (gt_lab == fdi).astype(np.uint8)
        ys, xs = np.nonzero(m)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea).reshape(-1, 2)
        teeth.append({"fdi": str(int(fdi)),                  # dataset stores FDI as text
                      "centroid": {"x": float(xs.mean()), "y": float(ys.mean())},
                      "contour": c.tolist()})
    import json
    with open(os.path.join(folder, "tooth_contours.json"), "w") as f:
        json.dump({"patient_id": "SYN", "case_name": "syn", "teeth": teeth}, f)


def gui_tests(tmp):
    ok = True
    gui = FakeGUI()
    gui.install()
    try:
        world = make_arch(True)
        folder = os.path.join(tmp, "gui_case")
        save_scanner_frame(world, folder, 11)
        photo, gt_lab, gt_ref = photograph(dict(world), True, seed=2)
        H, W = photo.shape[:2]
        ps = min(1.0, do.MAX_DISPLAY / W, do.MAX_DISPLAY / H)
        teeth = do.prepare_teeth(do.load_case(folder))
        frame = do.anatomical_frame(teeth)
        order = do.click_order(list(teeth))
        gv = visible_centres(gt_lab)
        P = lambda f, dx=0, dy=0: ((gv[f][0] + dx) * ps, (gv[f][1] + dy) * ps)

        print("Guided clicking:")
        guide = do.Guide(teeth, frame)
        sc = [click(*P(11)), ("key", ord("m")), click(*P(21)), ("key", ord("m")),
              click(*P(21, 90, 0)),                 # tooth 12 clicked on the WRONG side
              click(*P(12)), click(*P(22)), click(*P(13)),
              click(*P(23, 0, 260)),                # far away (lower row) -> rejected
              ("key", ord("f")),                    # force-keep it ...
              ("key", ord("u")),                    # ... then undo it
              click(*P(23)),
              ("key", ord("a")),                    # accept predicted spot for tooth 14
              ]
        for f in order[order.index(24):]:
            sc.append(click(*P(f)))
        sc.append(("key", 13))
        gui.script = sc
        clicks = do.collect_clicks(photo, guide)
        ok &= check("all 12 teeth recorded", len(clicks) == 12, f"(got {len(clicks)})")
        ok &= check("wrong-side click for 12 was refused",
                    abs(clicks[12][0] - gv[12][0]) < 2, f"x={clicks[12][0]:.0f} vs {gv[12][0]:.0f}")
        ok &= check("far click for 23 was undone / replaced",
                    abs(clicks[23][1] - gv[23][1]) < 2)
        err14 = np.hypot(*(np.array(clicks[14]) - gt_ref[14]))
        s_px = 12.0
        ok &= check("'accept expected spot' lands on tooth 14", err14 < 8 * s_px, f"({err14/s_px:.1f} mm)")

        print("Fine-tune window (drag a tooth, per-tooth snap):")
        sess = do.Session(teeth, frame, photo, {f: tuple(gv[f]) for f in gv if f % 10 <= 6},
                          folder, flip_u=guide.flip_u, flip_v=guide.flip_v)
        sw, sh = int(W * ps), int(H * ps)
        gui.tb.clear()
        v0 = sess.renderer.render(sess.pose_for(), sw, sh, ps)["ref_xy"][13]
        gui.script = [("key", 255),
                      ("mouse", cv2.EVENT_LBUTTONDOWN, v0[0], v0[1], 0),
                      ("mouse", cv2.EVENT_MOUSEMOVE, v0[0] + 20, v0[1], cv2.EVENT_FLAG_LBUTTON),
                      ("mouse", cv2.EVENT_MOUSEMOVE, v0[0] + 40, v0[1], cv2.EVENT_FLAG_LBUTTON),
                      ("mouse", cv2.EVENT_LBUTTONUP, v0[0] + 40, v0[1], 0),
                      ("key", 255),
                      ("key", ord("o")), ("key", ord("l")), ("key", ord("c")), ("key", ord("c")),
                      ("key", ord("t")), ("key", ord("t")), ("key", ord("h")), ("key", ord("h")),
                      ("key", ord("v")), ("key", ord("v")), ("key", ord("r")),
                      ("key", ord("s")), ("key", ord("q"))]
        before = sess.clicks[13][0]
        do.run_gui(sess, guide)
        ok &= check("dragging moved tooth 13 by ~40 preview px",
                    abs((sess.clicks[13][0] - before) * ps - 40) < 1.5,
                    f"({(sess.clicks[13][0]-before)*ps:.1f})")
        ok &= check("save wrote overlay + rgba + pose files",
                    all(os.path.exists(f) for f in ("final_overlay.png", "final_overlay_rgba.png", "final_pose.json")))

        print("Redo clicks (x key):")
        gui.tb.clear()
        redo = [click(*P(f)) for f in order] + [("key", 13)]
        gui.script = [("key", 255), ("key", ord("x"))] + redo + [("key", 255), ("key", ord("q"))]
        sess.clicks = {f: (gv[f][0] + 40, gv[f][1]) for f in order}      # deliberately bad
        sess.rebuild()
        do.run_gui(sess, guide)
        e = np.mean([np.hypot(*(np.array(sess.clicks[f]) - gv[f])) for f in order])
        ok &= check("redo replaced the clicks", e < 3, f"(mean err {e:.1f}px)")

        print("Dataset mode (annotated photo, no clicking):")
        pf = os.path.join(tmp, "Patient_9")
        make_gt_patient_folder(pf, photo, gt_lab)
        with open("patient_index.json", "w") as fh:
            import json
            json.dump({"Patient_9": pf}, fh)
        pp, cents, conts = do.load_patient("9")                     # resolved through the index
        ph2 = do.load_photo(pp)
        cl = {f: p for f, p in cents.items() if f in teeth and f % 10 <= 6}
        # other person: model teeth differ from the photo's
        sess2 = do.Session(teeth, frame, ph2, cl, folder, contours={f: c for f, c in conts.items() if f in teeth})
        res = {}
        for sc_on in (False, True):
            sess2.scales_on = sc_on
            rend, _ = sess2.render_full(sess2.pose_for(), 0.5)
            res[sc_on] = iou(rend["lab"] > 0, gt_lab > 0)
        ok &= check("dataset mode IoU high", res[False] > 0.90, f"(default {res[False]:.2f}; optional per-tooth size {res[True]:.2f})")
    finally:
        gui.restore()
    return ok


def two_jaw_test(tmp):
    import json
    print("Both jaws:")
    up = make_arch(True)
    lo_raw = make_arch(False, 24.0, 20.0)                       # lower arch: smaller, crowns point up
    lo = {f: (V + np.array([0.0, -3.0, -13.0]), F) for f, (V, F) in lo_raw.items()}   # behind + below, ~2 mm overbite
    merged = dict(up)
    merged.update(lo)
    photo, gt_lab, gt_ref = photograph(dict(merged), True, seed=2)
    fu, fl = os.path.join(tmp, "jaw_u"), os.path.join(tmp, "jaw_l")
    save_scanner_frame(up, fu, 31)
    save_scanner_frame(lo, fl, 32)
    gv = visible_centres(gt_lab)
    cu = {f: tuple(gv[f]) for f in gv if f // 10 in (1, 2) and f % 10 <= 6}
    cl = {f: tuple(gv[f]) for f in gv if f // 10 in (3, 4) and f % 10 <= 6}
    tu = do.prepare_teeth(do.load_case(fu)); tl = do.prepare_teeth(do.load_case(fl))
    fru, frl = do.anatomical_frame(tu), do.anatomical_frame(tl)
    su = do.Session(tu, fru, photo, cu, fu)
    sl = do.Session(tl, frl, photo, cl, fl)
    ok = check("jaw types detected", do.jaw_name(su) == "upper" and do.jaw_name(sl) == "lower")
    H, W = photo.shape[:2]
    comb = np.zeros((H, W), np.uint8)
    for s in (sl, su):                                          # lower first, upper on top
        rend = s.renderer.render(s.pose_for(), W, H, 1.0)
        inv = {v: k for k, v in s.renderer.label_id.items()}
        m = rend["lab"] > 0
        comb[m] = np.vectorize(lambda i: inv[i])(rend["lab"][m])
    for name, quads in (("upper", (1, 2)), ("lower", (3, 4))):
        v = np.mean([iou(comb == f, gt_lab == f) for f in gt_ref if f // 10 in quads and f % 10 <= 6])
        ok &= check(f"{name} jaw overlaps its own teeth (both drawn together)", v > 0.6, f"(per-tooth IoU {v:.2f})")

    gui = FakeGUI()
    gui.install()
    try:
        gu, gl = do.Guide(tu, fru), do.Guide(tl, frl)
        gu.flip_u, gu.flip_v, gl.flip_u, gl.flip_v = su.flip_u, su.flip_v, sl.flip_u, sl.flip_v
        gu.rebuild(); gl.rebuild()
        ps = min(1.0, do.MAX_DISPLAY / W, do.MAX_DISPLAY / H)
        sw, sh = int(W * ps), int(H * ps)
        seen = {}
        setbar = lambda n, v: (lambda: gui.tb.__setitem__(n, v))
        note = lambda key: (lambda: seen.__setitem__(key, gui.tb["Scale %"]))
        r41 = sl.renderer.render(sl.pose_for(), sw, sh, ps)["ref_xy"][41]
        before41 = sl.clicks[41]
        gui.tb.clear()
        gui.script = [("key", 255),
                      ("call", setbar("Scale %", 120)), ("key", 255),          # upper jaw -> 120 %
                      ("key", ord("j")), ("call", note("lower_default")),       # lower jaw starts at 100 %
                      ("call", setbar("Scale %", 80)), ("key", 255),            # lower -> 80 %
                      ("mouse", cv2.EVENT_LBUTTONDOWN, r41[0], r41[1], 0),       # drag a LOWER tooth
                      ("mouse", cv2.EVENT_MOUSEMOVE, r41[0] + 30, r41[1], cv2.EVENT_FLAG_LBUTTON),
                      ("mouse", cv2.EVENT_LBUTTONUP, r41[0] + 30, r41[1], 0),
                      ("key", ord("j")), ("call", note("upper_back")),          # upper kept its 120 %
                      ("key", ord("s")), ("key", ord("q"))]
        do.run_gui(su, gu, extra=[(sl, gl)])
        ok &= check("each jaw keeps its own slider values", seen.get("lower_default") == 100 and seen.get("upper_back") == 120,
                    f"(lower started at {seen.get('lower_default')}, upper back at {seen.get('upper_back')})")
        ok &= check("dragging a lower tooth moved only that tooth",
                    abs((sl.clicks[41][0] - before41[0]) * ps - 30) < 1.5)
        data = json.load(open("final_pose.json"))
        ok &= check("combined save has both jaws + all clicks",
                    set(data.get("jaws", {})) == {"upper", "lower"} and len(data["clicks"]) == len(cu) + len(cl))
        ok &= check("saved poses differ per jaw (1.2 vs 0.8)",
                    abs(data["jaws"]["upper"]["pose"]["g_scale"] - 1.2) < 1e-6
                    and abs(data["jaws"]["lower"]["pose"]["g_scale"] - 0.8) < 1e-6)
        img = cv2.imread("final_overlay.png"); rgba = cv2.imread("final_overlay_rgba.png", cv2.IMREAD_UNCHANGED)
        ok &= check("combined overlay images written", img.shape[:2] == (H, W) and rgba.shape == (H, W, 4))
        # skipping the second jaw while clicking
        order = do.click_order(list(tl))
        gui.script = [("key", 255), ("key", ord("k"))]
        res = do.collect_clicks(photo, gl, allow_skip=True)
        ok &= check("'k' skips a jaw", res == {})
    finally:
        gui.restore()
    # the real command-line entry point, both jaws, in dataset mode and in saved-clicks mode
    import sys as _sys
    gui2 = FakeGUI()
    gui2.install()
    try:
        pf = os.path.join(tmp, "Patient_both")
        make_gt_patient_folder(pf, photo, gt_lab)
        cv2.imwrite("p.jpg", photo)
        for label, argv in (("--patient", ["--patient", pf]),
                            ("--clicks", ["--clicks", "final_pose.json", "--photo", "p.jpg"])):
            for f_ in ("final_pose.json",) if label == "--patient" else ():
                if os.path.exists(f_):
                    os.remove(f_)
            if label == "--clicks":
                json.dump({"clicks": {str(f): list(map(float, gv[f])) for f in gv if f % 10 <= 6}},
                          open("final_pose.json", "w"))
            old_argv = _sys.argv
            _sys.argv = ["dental_overlay.py", "--case", fu, "--case-lower", fl] + argv
            gui2.tb.clear()
            gui2.script = [("key", 255), ("key", ord("s")), ("key", ord("q"))]
            try:
                do.main()
            finally:
                _sys.argv = old_argv
            d = json.load(open("final_pose.json"))
            ok &= check(f"main() {label}: both jaws shown and saved", set(d.get("jaws", {})) == {"upper", "lower"})
    finally:
        gui2.restore()
    # --both when the lower library has NO folder with the upper case's ID: any lower case is used
    up2 = os.path.join(tmp, "alt", "processed_part1_upper", "AAA")
    lo2 = os.path.join(tmp, "alt", "processed_part1_lower", "ZZZ")
    save_scanner_frame(up, up2, 33)
    save_scanner_frame(lo, lo2, 34)
    path, how = do.find_lower(up2)
    ok &= check("lower arch found even when the case IDs differ", path == lo2 and "automatically" in how, f"({how})")
    gui3 = FakeGUI()
    gui3.install()
    try:
        old_argv = _sys.argv
        _sys.argv = ["dental_overlay.py", "--case", up2, "--both", "--patient", pf]
        gui3.tb.clear()
        gui3.script = [("key", 255), ("key", ord("s")), ("key", ord("q"))]
        try:
            do.main()
        finally:
            _sys.argv = old_argv
        d = json.load(open("final_pose.json"))
        ok &= check("main() --both with differing IDs shows both jaws", set(d.get("jaws", {})) == {"upper", "lower"})
    finally:
        gui3.restore()
    # --both with nothing to find fails loudly instead of silently showing one jaw
    try:
        _sys.argv = ["dental_overlay.py", "--case", os.path.join(tmp, "jaw_u"), "--both", "--patient", pf]
        do.main()
        ok &= check("--both with no lower folder exits with a clear message", False)
    except SystemExit as e:
        ok &= check("--both with no lower folder exits with a clear message", "check_lower" in str(e))
    finally:
        _sys.argv = old_argv
    # the lower folder is found next to the upper one
    up_dir = os.path.join(tmp, "processed_part1_upper", "CASEX"); lo_dir = os.path.join(tmp, "processed_part1_lower", "CASEX")
    os.makedirs(up_dir); os.makedirs(lo_dir)
    ok &= check("matching lower folder derived from the upper path",
                do.derive_lower(up_dir) == lo_dir and do.derive_lower(os.path.join(tmp, "nowhere", "x")) is None)
    return ok


def library_test(tmp):
    import find_best_arch as fb
    print("Arch finder + saved-clicks reuse:")
    base = (WIDTHS[:], HEIGHTS[:], DEPTHS[:])

    def person(seed):
        rng = np.random.default_rng(seed)
        WIDTHS[:] = [w * rng.uniform(.88, 1.12) for w in base[0]]
        HEIGHTS[:] = [h * rng.uniform(.8, 1.2) for h in base[1]]
        DEPTHS[:] = [d * rng.uniform(.9, 1.1) for d in base[2]]
        w = make_arch(True, 27 * rng.uniform(.92, 1.08), 23 * rng.uniform(.9, 1.1))
        WIDTHS[:], HEIGHTS[:], DEPTHS[:] = base
        return w

    n_lib = 24
    lib = os.path.join(tmp, "library")
    for i in range(n_lib):
        save_scanner_frame(person(500 + i), os.path.join(lib, f"CASE{i:02d}"), i)
    os.makedirs(os.path.join(lib, "not_a_case"), exist_ok=True)
    cases = fb.list_cases(lib)
    cache = fb.build_cache(cases, {}, 100, 1)
    good = check(f"library indexed ({n_lib} cases, junk folder ignored)",
                 len(cases) == n_lib and sum("refs" in v for v in cache.values()) == n_lib)

    def per_tooth(folder, photo, clicks, gt_lab):
        t = do.prepare_teeth(do.load_case(folder))
        s = do.Session(t, do.anatomical_frame(t), photo, clicks, folder)
        rend, _ = s.render_full(s.pose_for(), 0.5)
        return np.mean([iou(rend["lab"] == s.renderer.label_id[f], gt_lab == f)
                        for f in s.cam if f % 10 <= 6])

    picked, typical, last = [], [], None
    for seed in range(5):                     # 5 people who are NOT in the library
        photo, gt_lab, _ = photograph(dict(person(seed)), True, seed=seed)
        gv = visible_centres(gt_lab)
        clicks = {f: tuple(gv[f]) for f in gv if f % 10 <= 6}
        rows = fb.rank(cache, clicks)
        picked.append(per_tooth(rows[0][1], photo, clicks, gt_lab))
        typical.append(np.mean([per_tooth(os.path.join(lib, f"CASE{j:02d}"), photo, clicks, gt_lab)
                                for j in (3, 11, 19)]))
        last = (photo, clicks, rows)
    good &= check("auto-picked arch overlaps better than a typical arch",
                  np.mean(picked) > np.mean(typical),
                  f"(typical {np.mean(typical):.2f} -> picked {np.mean(picked):.2f})")
    photo, clicks, rows = last
    good &= check("ranking is sorted best-first", all(rows[i][0] <= rows[i + 1][0] for i in range(len(rows) - 1)))
    teeth = do.prepare_teeth(do.load_case(rows[0][1]))
    frame = do.anatomical_frame(teeth)
    sess = do.Session(teeth, frame, photo, clicks, rows[0][1])
    do.save_results(sess, sess.pose_for(), 0.5, tag="rt")
    back = do.load_clicks_file("rt_pose.json")
    good &= check("saved clicks reload identically",
                  set(back) == set(clicks) and all(np.allclose(back[f], clicks[f]) for f in clicks))
    return good


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    allok = True
    first = None
    print("Geometry / fit (independent camera, noisy clicks):")
    for upper in (True, False):
        for seed in (1, 2, 3):
            r = run_case(upper, seed, tmp)
            p = r["pose"]
            print(f"  {'UPPER' if upper else 'LOWER'} seed {seed}: fit {p['rms_mm']:.2f} mm | "
                  f"IoU no-snap {r[False]['iou']:.2f}, snap {r[True]['iou']:.2f} | "
                  f"render {r[True]['secs']*1000:.0f} ms (full-res)")
            allok &= r[False]["iou"] > 0.70 and not r["flip_u"]
            if first is None:
                first = r
                cv2.imwrite("selftest_photo.png", r["photo"])
                cv2.imwrite("selftest_overlay_snap.png", r[True]["img"])
    work = tempfile.mkdtemp()
    old = os.getcwd()
    os.chdir(work)
    try:
        allok &= gui_tests(tmp)
        allok &= library_test(tmp)
        allok &= two_jaw_test(tmp)
    finally:
        os.chdir(old)
    print("\nSELF-TEST", "PASSED" if allok else "FAILED")
    sys.exit(0 if allok else 1)