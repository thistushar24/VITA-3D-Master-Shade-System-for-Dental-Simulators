"""Regenerate the SYNTHETIC sample photos/annotations in data/samples from the procedural (licence-free) models.
Run build_demo_library.py first. None of these are real dental photographs."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dental_reg import core, library
from dental_reg.synth import render

data = Path(__file__).resolve().parents[1] / "data"
out, lib = data / "samples", data / "demo_library"
W, H = 1280, 853
K = core.intrinsics(W, H)


def pose(arch, K, w, h, euler, shift):
    b = core.initial_pose(arch, w, h, K)
    return core.Pose((Rot.from_euler("xyz", euler, degrees=True) * Rot.from_rotvec(b.rvec)).as_rotvec(), b.t + shift, 1.0, b.center)


# ---- registration test sample: per-tooth OBJ case folder -> grey-white render + mask + landmarks + ground truth
arch = core.load_arch_dir(out / "proc_upper_01")
gt = pose(arch, K, W, H, [5, 8, 3], [6, -4, 10])
img, mask = render(arch, gt, K, W, H)
cv2.imwrite(str(out / "synthetic_photo.png"), img)
cv2.imwrite(str(out / "synthetic_mask.png"), mask)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
poly = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 2.0, True)[:, 0, :]
lms = [{"fdi": f, "image_xy": core.project(arch.teeth[f][None], gt, K)[0][0].round(1).tolist()} for f in (11, 21, 13, 23, 16, 26)]
(out / "synthetic_annotations.json").write_text(json.dumps({"_note": "SYNTHETIC render of proc_upper_01", "markers": [],
                                                           "landmarks": lms, "mask_polygons": [poly.tolist()]}, indent=1))
(out / "synthetic_ground_truth.json").write_text(json.dumps({"pose": gt.to_json(), "camera_K": K.tolist(), "image_size": [W, H]}, indent=1))

# ---- v1 demo pair: Photo A (retractor-style) and Photo B (flat 'face'), from library model upper_proc04
src = library.load_arch_npz(lib / "upper_proc04.npz")
pa = pose(src, K, W, H, [3, 6, 2], [5, -3, 8])
a_img, _ = render(src, pa, K, W, H, tooth=(170, 215, 240), bg=(40, 30, 150))
cv2.imwrite(str(out / "demo_photo_a.png"), a_img)
WB, HB = 1000, 700
Kb = core.intrinsics(WB, HB, 45)
pb = pose(src, Kb, WB, HB, [-2, -4, 3], [-10, 25, 120])
face = np.full((HB, WB, 3), (150, 175, 225), np.uint8)
cv2.ellipse(face, (WB // 2 - 15, HB // 2 + 40), (210, 80), 0, 0, 360, (60, 50, 120), -1)
b_img, _ = render(src, pb, Kb, WB, HB, tooth=(205, 225, 240), bg=(0, 0, 0))
m = b_img.sum(2) > 0
face[m] = b_img[m]
cv2.imwrite(str(out / "demo_photo_b.png"), face)
(out / "demo_annotations.json").write_text(json.dumps({
    "_note": "SYNTHETIC demo: canine points on the model's true canines in demo_photo_a.png (stand-in for the user dragging them)",
    "canines": [core.project(src.teeth[f][None], pa, K)[0][0].round(1).tolist() for f in (13, 23)]}))
print("wrote synthetic samples")
