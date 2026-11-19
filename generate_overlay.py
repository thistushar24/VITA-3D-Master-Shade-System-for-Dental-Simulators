"""
Phase 3 - The actual visual deliverable: a transparent 3D arch overlaid
on the patient's photo, positioned using the transform computed in
auto_align.py.

auto_align.py only calculated WHERE the model should go. This script
actually draws every tooth from the 3D model onto the photo using that
transform - not just the teeth that were used to calculate it.
"""

import cv2
import numpy as np
from contour_loader import load_2d_annotations
from build_patient_index import load_index
from auto_align import get_3d_tooth_centroids, compute_initial_alignment


def flatten_arch_to_image(loader, canvas_scale=10, padding=20):
    """
    Projects every tooth mesh's vertices straight down onto the XY plane
    (ignoring depth) and draws filled tooth silhouettes onto one flat image.
    This includes ALL loaded teeth, regardless of whether they had a match
    in the 2D photo annotation.

    Returns:
        flat_img: BGRA image of the whole arch silhouette, transparent background
        min_x, min_y: the model-space point this image's (0,0) corresponds to
        canvas_scale: pixels drawn per 1 model unit (needed to undo later)
    """
    all_points = []
    tooth_point_lists = []

    for fdi, mesh in loader.tooth_meshes.items():
        pts = np.asarray(mesh.vertices)[:, :2]  # drop Z, keep X, Y
        tooth_point_lists.append(pts)
        all_points.append(pts)

    all_points = np.vstack(all_points)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)

    width = int((max_x - min_x) * canvas_scale) + padding * 2
    height = int((max_y - min_y) * canvas_scale) + padding * 2

    flat_img = np.zeros((height, width, 4), dtype=np.uint8)  # transparent BGRA

    for pts in tooth_point_lists:
        px = ((pts[:, 0] - min_x) * canvas_scale + padding).astype(np.int32)
        py = ((pts[:, 1] - min_y) * canvas_scale + padding).astype(np.int32)
        hull = cv2.convexHull(np.stack([px, py], axis=1))
        cv2.fillConvexPoly(flat_img, hull, (60, 220, 80, 190))  # semi-transparent green

    return flat_img, min_x, min_y, canvas_scale, padding


def composite_overlay(photo_path, flat_img, min_x, min_y, canvas_scale, padding, transform_matrix):
    """
    Warps the flat arch silhouette into photo pixel space using the
    computed transform, then alpha-blends it onto the real photo.
    """
    photo = cv2.imread(photo_path)
    ph, pw = photo.shape[:2]

    # "Flat-image-pixel space" -> "model space" (undoes BOTH the drawing
    # scale AND the padding offset applied when the silhouette was drawn)
    S = np.array([
        [1.0 / canvas_scale, 0, min_x - padding / canvas_scale],
        [0, 1.0 / canvas_scale, min_y - padding / canvas_scale],
        [0, 0, 1],
    ])

    # "model space" -> "photo pixel space" (the transform auto_align computed)
    M = np.vstack([transform_matrix, [0, 0, 1]])

    # Combined: flat-image-pixel space -> photo pixel space, in one step
    T = (M @ S)[:2, :]

    warped = cv2.warpAffine(flat_img, T, (pw, ph))

    opacity = 0.55
    alpha = (warped[:, :, 3] / 255.0) * opacity
    canvas = photo.astype(np.float32)
    for c in range(3):
        canvas[:, :, c] = alpha * warped[:, :, c] + (1 - alpha) * canvas[:, :, c]

    return canvas.astype(np.uint8)


if __name__ == "__main__":
    CASE_DIR = r"C:\Users\utkar\Downloads\DENTAL\VITA-3D-Master-Shade-System-for-Dental-Simulators\processed_part1_upper\M7RTPNPC"
    PATIENT_ID = "Patient_1"

    patient_index = load_index()
    PATIENT_DIR = patient_index[PATIENT_ID]

    centroids_3d, loader = get_3d_tooth_centroids(CASE_DIR)
    photo_path, annotations_2d = load_2d_annotations(PATIENT_DIR)

    # Step 1: calculate the transform (same as before)
    transform_matrix, common_fdi, src_pts, dst_pts = compute_initial_alignment(
        centroids_3d, annotations_2d
    )

    # Step 2: draw ALL teeth from the 3D model, not just the matched ones
    flat_img, min_x, min_y, canvas_scale, padding = flatten_arch_to_image(loader)

    # Step 3: place that drawing onto the real photo using the transform
    result = composite_overlay(photo_path, flat_img, min_x, min_y, canvas_scale, padding, transform_matrix)

    cv2.imshow("3D arch overlaid on photo - press any key to close", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("auto_overlay_result.png", result)
    print("Saved result to auto_overlay_result.png")