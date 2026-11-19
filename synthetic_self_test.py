"""
Synthetic self-test for the alignment + overlay pipeline.

We don't yet have a real photo + 3D model pair for the SAME patient, so we
can't judge visual correctness from real data. This script creates a FAKE
"photo" by taking your real 3D model's own tooth centroids and placing them
using a transform WE choose (so we know the exact right answer in advance).

If the code correctly recovers that same transform and draws a clean,
correctly-positioned arch, the pipeline's math is proven correct - any
strange-looking result from real mismatched patients is a data problem,
not a code problem.
"""

import json
import os
import numpy as np
import cv2
from auto_align import get_3d_tooth_centroids, compute_initial_alignment
from generate_overlay import flatten_arch_to_image, composite_overlay
from contour_loader import load_2d_annotations


def make_synthetic_patient(case_dir, out_dir, known_scale=4.0, known_angle_deg=5.0,
                            known_tx=400, known_ty=300, canvas_size=(800, 600)):
    """
    Builds a fake patient folder (original.png + tooth_contours.json) by
    applying a KNOWN transform to the real 3D model's own tooth centroids.
    """
    centroids_3d, loader = get_3d_tooth_centroids(case_dir)

    angle_rad = np.deg2rad(known_angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    teeth_entries = []
    for fdi, (x, y, z) in centroids_3d.items():
        px = known_scale * (cos_a * x - sin_a * y) + known_tx
        py = known_scale * (sin_a * x + cos_a * y) + known_ty
        teeth_entries.append({
            "fdi": fdi,
            "centroid": {"x": float(px), "y": float(py)},
            "contour": [[float(px), float(py)]],  # placeholder - not used yet
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tooth_contours.json"), "w") as f:
        json.dump(
            {"patient_id": "SYNTHETIC", "case_name": "synthetic_test", "teeth": teeth_entries},
            f, indent=2,
        )

    blank = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
    cv2.imwrite(os.path.join(out_dir, "original.png"), blank)

    print(f"Synthetic patient created at: {out_dir}")
    print(f"Known (ground-truth) transform: scale={known_scale}, "
          f"angle={known_angle_deg} deg, tx={known_tx}, ty={known_ty}")
    return loader, centroids_3d


if __name__ == "__main__":
    CASE_DIR = r"C:\Users\utkar\Downloads\DENTAL\VITA-3D-Master-Shade-System-for-Dental-Simulators\processed_part1_upper\M7RTPNPC"
    SYNTH_DIR = "synthetic_patient"

    loader, centroids_3d = make_synthetic_patient(CASE_DIR, SYNTH_DIR)

    photo_path, annotations_2d = load_2d_annotations(SYNTH_DIR)

    transform_matrix, common_fdi, src_pts, dst_pts = compute_initial_alignment(
        centroids_3d, annotations_2d
    )
    print("Recovered transform (should closely match the known one above):\n", transform_matrix)

    flat_img, min_x, min_y, canvas_scale, padding = flatten_arch_to_image(loader)
    result = composite_overlay(photo_path, flat_img, min_x, min_y, canvas_scale, padding, transform_matrix)

    cv2.imshow("Synthetic self-test - should show a clean, correctly sized arch shape", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("synthetic_test_result.png", result)
    print("Saved to synthetic_test_result.png")