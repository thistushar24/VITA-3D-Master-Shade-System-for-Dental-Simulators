"""Benchmark canine-point estimation on synthetic renders of a model library (true canines are known).
usage: python bench_canines.py LIBRARY_DIR [N]   -- prints errors as a fraction of the true canine distance."""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dental_reg import core, library, vision
from dental_reg.synth import render

lib = sys.argv[1]
cat = [e for e in library.load_catalogue(lib) if e["jaw"] == "upper" and e["classification"] == "complete_arch"][: int(sys.argv[2]) if len(sys.argv) > 2 else 40]
W, H = 1280, 853
K = core.intrinsics(W, H)
rng = np.random.default_rng(0)
errs, fails, methods, ratios, ex, ey, bym = [], 0, {}, [], [], [], {}
for e in cat:
    a = library.load_arch_npz(f"{lib}/{e['file']}")
    b = core.initial_pose(a, W, H, K)
    p = core.Pose((Rot.from_euler("xyz", rng.uniform(-6, 6, 3), degrees=True) * Rot.from_rotvec(b.rvec)).as_rotvec(),
                  b.t + rng.uniform(-8, 8, 3) * [1, 1, 3], 1.0, b.center)
    img, mask = render(a, p, K, W, H, tooth=(170, 215, 240), bg=(40, 30, 150))
    true = core.project(np.array([a.teeth[13], a.teeth[23]]), p, K)[0]
    ys, xs = np.nonzero(mask)
    ratios.append(np.linalg.norm(true[0] - true[1]) / (xs.max() - xs.min() + 1))
    try:
        seg = vision.segment_teeth(img)
    except core.InputError:
        fails += 1
        continue
    methods[seg["canine_method"]] = methods.get(seg["canine_method"], 0) + 1
    cd = np.linalg.norm(true[0] - true[1])
    d = np.array(seg["canines"]) - true
    errs += list(np.linalg.norm(d, axis=1) / cd)
    ex += list(np.abs(d[:, 0]) / cd); ey += list(np.abs(d[:, 1]) / cd)
    bym.setdefault(seg["canine_method"], []).extend(np.linalg.norm(d, axis=1) / cd)
print(f"models={len(cat)} segmentation_failures={fails} methods={methods}")
print(f"canine distance / arch width: median {np.median(ratios):.3f} (min {min(ratios):.3f}, max {max(ratios):.3f})")
print(f"error/canine_distance: median {np.median(errs):.3f}  p90 {np.percentile(errs, 90):.3f}  max {max(errs):.3f}  (n={len(errs)})")
print(f"median |dx|/cd {np.median(ex):.3f}  median |dy|/cd {np.median(ey):.3f}")
for k, v in bym.items():
    print(f"  {k}: median {np.median(v):.3f} p90 {np.percentile(v, 90):.3f} n={len(v)}")
