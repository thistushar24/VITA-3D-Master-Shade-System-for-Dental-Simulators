"""Automatic best-fit model ranking for Photo A.

Inputs: upper-arch tooth mask, two canine image points (patient's right = image-left = FDI 13, patient's left = FDI 23),
the model catalogue. Shade is deliberately NOT used. Output is a reproducible ranked list with all feature values.

Score (documented, hand-set weights; NOT a trained model and NOT a probability/accuracy):
  stage 1 (all candidates, cheap):  s1 = 0.6 * shapeIoU + 0.4 * canine_ratio_score
     shapeIoU: IoU of the frontal model silhouette and the photo mask, both cropped to their bounding box and resized
               to 96x48 (translation/scale invariant: compares aspect ratio + tooth arrangement)
     canine_ratio_score = max(0, 1 - |r_model - r_photo| / r_photo),  r = canine distance / silhouette width
  stage 2 (top K by s1, full registration with the two canine points as landmark residuals):
     s2 = 0.5 * IoU_fit + 0.3 * canine_fit + 0.2 * s1
     canine_fit = max(0, 1 - rmse_px / (0.08 * mask_width_px))
Candidates lacking canines/central incisors (partial arches) are ranked after complete ones.
"""
from __future__ import annotations

import numpy as np
import cv2

from . import core, library
from .core import InputError

SHAPE_SIZE = (96, 48)
PRELIM_BELOW = 0.7  # hand-set: below this absolute final score the match is flagged preliminary
WEIGHTS = {"s1_shape": 0.6, "s1_canine": 0.4, "s2_iou": 0.5, "s2_canine": 0.3, "s2_s1": 0.2}


def _norm_crop(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    c = (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(np.uint8) * 255
    return cv2.resize(c, SHAPE_SIZE, interpolation=cv2.INTER_AREA) > 127, (xs.max() - xs.min() + 1)


def _canine_ratio(px, width):
    return float(np.hypot(px[0][0] - px[1][0], px[0][1] - px[1][1]) / max(width, 1))


def rank_models(mask_upper, canines, catalogue, lib_dir, img_shape, fov_deg=40.0, top_k=6, jaw="upper", loader=None):
    h, w = img_shape[:2]
    if mask_upper is None or (mask_upper > 0).sum() < 100:
        raise InputError("Matching needs a tooth region (automatic segmentation or a drawn outline)")
    if canines is None or len(canines) != 2:
        raise InputError("Matching needs the two canine reference points")
    loader = loader or (lambda e: library.load_arch_npz(f"{lib_dir}/{e['file']}"))
    K = core.intrinsics(w, h, fov_deg)
    photo, photo_w = _norm_crop(mask_upper)
    r_photo = _canine_ratio(canines, photo_w)
    cands = [e for e in catalogue if e["jaw"] == jaw]
    if not cands:
        raise InputError(f"The model catalogue has no {jaw}-arch models")
    rows = []
    for e in cands:
        a = loader(e)
        base = core.initial_pose(a, w, h, K)
        sil = core.silhouette(a.vertices, a.faces, base, K, w, h, a.face_tooth)
        sh = _norm_crop(sil)
        if sh is None or 13 not in a.teeth or 23 not in a.teeth:
            canine_score, shape_iou, r_model = 0.0, 0.0, None
            if sh is not None:
                shape_iou = float(np.logical_and(sh[0], photo).sum() / np.logical_or(sh[0], photo).sum())
        else:
            uv, _, _ = core.project(np.array([a.teeth[13], a.teeth[23]]), base, K)
            r_model = _canine_ratio(uv, sh[1])
            canine_score = max(0.0, 1 - abs(r_model - r_photo) / r_photo)
            shape_iou = float(np.logical_and(sh[0], photo).sum() / np.logical_or(sh[0], photo).sum())
        s1 = WEIGHTS["s1_shape"] * shape_iou + WEIGHTS["s1_canine"] * canine_score
        rows.append({"id": e["id"], "classification": e["classification"], "shape_iou": shape_iou,
                     "canine_ratio_model": r_model, "canine_ratio_score": canine_score, "stage1": s1})
    rows.sort(key=lambda r: (r["classification"] != "complete_arch", -r["stage1"], r["id"]))
    cfg = core.RegConfig(polish_widths=(320,), max_nfev=80)
    by_id = {e["id"]: e for e in cands}
    for r in rows[:top_k]:
        a = loader(by_id[r["id"]])
        lm = [(13, tuple(canines[0])), (23, tuple(canines[1]))] if 13 in a.teeth and 23 in a.teeth else []
        try:
            pose, rep = core.register(a, img_shape, K, landmarks=lm, mask=mask_upper, cfg=cfg)
            iou = rep["metrics"]["silhouette_iou"]
            rm = rep["metrics"].get("landmark_rmse_px")
        except InputError:
            iou, rm = 0.0, None
        cf = 0.0 if rm is None else max(0.0, 1 - rm / (0.08 * photo_w))
        r.update(fit_iou=iou, canine_rmse_px=rm, canine_fit=cf,
                 final=WEIGHTS["s2_iou"] * iou + WEIGHTS["s2_canine"] * cf + WEIGHTS["s2_s1"] * r["stage1"])
    ranked = sorted([r for r in rows if "final" in r], key=lambda r: (r["classification"] != "complete_arch", -r["final"], r["id"]))
    rest = [r for r in rows if "final" not in r]
    best = ranked[0]
    margin = best["final"] - ranked[1]["final"] if len(ranked) > 1 else None
    prelim = best["final"] < PRELIM_BELOW or (margin is not None and margin < 0.02) or best["classification"] != "complete_arch"
    why = (f"{best['id']}: fitted silhouette IoU {best['fit_iou']:.2f}, canine reprojection error "
           f"{best['canine_rmse_px']:.0f}px, shape similarity {best['shape_iou']:.2f}." if best["canine_rmse_px"] is not None else
           f"{best['id']}: fitted silhouette IoU {best['fit_iou']:.2f}.")
    if prelim:
        why += " PRELIMINARY: low absolute score or near-tie with the runner-up; check or replace the model."
    return {"selected": best["id"], "preliminary": bool(prelim), "explanation": why, "margin": margin,
            "ranked": ranked + rest, "fov_deg": fov_deg, "weights": WEIGHTS, "n_candidates": len(cands),
            "note": "Scores are hand-weighted agreement measures with your photo/annotations, not accuracy or probability."}
