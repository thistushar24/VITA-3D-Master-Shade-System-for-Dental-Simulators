"""Photo A analysis: automatic tooth-region segmentation and estimated canine reference points.

Heuristic, colour based (no trained model): a bright, low-red seed (the central incisors) is grown with GrabCut inside a
window around it. Works on retractor photos with visible incisors; it can fail on unusual lighting/colour -
callers must treat the output as an editable proposal (`confidence` / `notes` say what was assumed).
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import InputError

UPPER_FRACTION = 0.75  # heuristic: upper arch = top 75% of the tooth-region height (lower incisors overlap the rest)
CANINE_FRAC = 0.33  # canine offset / mask width, from the median canine-distance/arch-width = 0.665 of 36 real arches


def segment_teeth(img: np.ndarray) -> dict:
    cv2.setRNGSeed(0)  # GrabCut uses k-means init: fix the seed for reproducible output
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    strict = ((L > 150) & (a < 22) & (b > 5) & (b < 60)).astype(np.uint8)
    strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, cl, st, cen = cv2.connectedComponentsWithStats(strict)
    cand = [(st[i, 4] * np.exp(-(((cen[i][0] - w / 2) / (0.25 * w)) ** 2 + ((cen[i][1] - 0.5 * h) / (0.2 * h)) ** 2)), i)
            for i in range(1, n) if st[i, 4] > 0.003 * w * h]
    if not cand:
        raise InputError("No tooth region found automatically (no bright, low-red area near the image centre). "
                         "Draw the tooth outline manually.")
    i = max(cand)[1]
    x, y, bw, bh = st[i, :4]
    cx, cy = x + bw / 2, y + bh / 2
    # neighbouring bright components in the same band belong to the same arch (incisors may not touch each other)
    near = [j for j in range(1, n) if st[j, 4] > 0.0005 * w * h and abs(cen[j][1] - cy) < 0.7 * bh and abs(cen[j][0] - cx) < 3.5 * bw]
    ux0, uy0 = min(st[j, 0] for j in near), min(st[j, 1] for j in near)
    ux1, uy1 = max(st[j, 0] + st[j, 2] for j in near), max(st[j, 1] + st[j, 3] for j in near)
    uw = ux1 - ux0
    x0, x1 = int(max(0, ux0 - 0.9 * uw)), int(min(w, ux1 + 0.9 * uw))
    y0, y1 = int(max(0, cy - 0.9 * bh)), int(min(h, cy + 0.8 * bh))
    seed_ids = near
    gm = np.full((h, w), cv2.GC_BGD, np.uint8)
    win = np.zeros((h, w), bool)
    win[y0:y1, x0:x1] = True
    gm[win] = cv2.GC_PR_BGD
    gm[win & (L > 105) & (a < 26) & (b > 2)] = cv2.GC_PR_FGD
    gm[cv2.erode(np.isin(cl, seed_ids).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
    cv2.grabCut(img, gm, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    m = ((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n2, cl2, st2, _ = cv2.connectedComponentsWithStats(m)
    m = (cl2 == 1 + int(np.argmax(st2[1:, 4]))).astype(np.uint8) * 255
    ys, xs = np.nonzero(m)
    if len(xs) < 0.01 * w * h:
        raise InputError("Automatic tooth region is too small to be reliable. Draw the tooth outline manually.")
    top, bot = np.percentile(ys, 1), np.percentile(ys, 99)
    cut = int(top + UPPER_FRACTION * (bot - top))
    upper = m.copy()
    upper[cut:] = 0
    canines, method = estimate_canines(upper)
    return {"mask_all": m, "mask_upper": upper, "midline_x": float((canines[0][0] + canines[1][0]) / 2), "canines": canines,
            "seed_bbox": [int(x), int(y), int(bw), int(bh)], "canine_method": method,
            "notes": ["Tooth region found by colour + GrabCut around the central incisors (heuristic).",
                      f"Upper arch = top {int(UPPER_FRACTION * 100)}% of the region (heuristic split from lower teeth).",
                      f"Canine points are ESTIMATED ({method}); drag them onto the real canines."],
            "confidence": "preliminary"}


def estimate_canines(upper: np.ndarray):
    """Canine reference points (image-left = patient's right = FDI 13, then FDI 23) from the upper-arch mask.

    Midline = mask bbox centre (symmetry assumption), offset = CANINE_FRAC x mask width, height = middle of the mask
    column at that x. (Using the central-incisor seed for the midline measured worse, see bench_canines.py.)
    """
    ys, xs = np.nonzero(upper)
    x0, x1 = xs.min(), xs.max()
    mid, off = (x0 + x1) / 2.0, CANINE_FRAC * float(x1 - x0 + 1)
    method = "symmetric about the mask centre, offset = 0.33 x mask width"
    pts = []
    for x in (mid - off, mid + off):
        x = float(np.clip(x, x0, x1))
        col = np.nonzero(upper[:, int(round(x))] > 0)[0]
        if len(col) == 0:  # column empty: use the mask's vertical middle
            col = ys
        pts.append([x, float((col.min() + col.max()) / 2.0)])
    return pts, method


def contours_from_mask(mask: np.ndarray, eps: float = 2.0) -> list:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) > 0.001 * mask.size]
    return [cv2.approxPolyDP(c, eps, True)[:, 0, :].tolist() for c in cs]
