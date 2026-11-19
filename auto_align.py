"""
Phase 3 - Automatic initial alignment using 2D-3D correspondence.

Both datasets label teeth by FDI number, so we can match specific teeth
between the 3D model and the 2D photo annotation (not just eyeball the
whole jaw into place). We use each matching tooth's center point as a
correspondence pair, then solve for the best-fit 2D transform
(rotation + scale + translation) that maps the 3D model onto the photo.

This gives a real starting point before any manual fine-tuning - far less
trial and error than dragging sliders from zero.
"""

import cv2
import numpy as np
from arch_loader import ArchLoader
from contour_loader import load_2d_annotations
from build_patient_index import load_index


def get_3d_tooth_centroids(case_dir):
    """Returns {fdi: (x, y, z)} center point of each individual tooth mesh."""
    loader = ArchLoader(case_dir)
    loader.load_teeth()

    centroids_3d = {}
    for fdi, mesh in loader.tooth_meshes.items():
        centroids_3d[fdi] = mesh.get_center()

    return centroids_3d, loader


def compute_initial_alignment(centroids_3d, annotations_2d):
    """
    Finds teeth present in BOTH the 3D model and the 2D photo (matched by
    FDI number), then solves for a similarity transform (rotation, uniform
    scale, translation - no shear) mapping 3D (x,y) -> 2D pixel (x,y).

    This assumes a roughly frontal/orthographic view, which is a reasonable
    approximation for a straight-on intraoral photo.
    """
    common_fdi = sorted(set(centroids_3d.keys()) & set(annotations_2d.keys()))
    print(f"Matching teeth found (present in both 3D and 2D): {common_fdi}")

    if len(common_fdi) < 3:
        raise ValueError(
            f"Need at least 3 matching teeth for a reliable transform, "
            f"found only {len(common_fdi)}"
        )

    # Source points: 3D model tooth centers (just X,Y - dropping depth)
    src_pts = np.array(
        [[centroids_3d[f][0], centroids_3d[f][1]] for f in common_fdi],
        dtype=np.float32,
    )
    # Destination points: real pixel centroids from the photo annotation
    dst_pts = np.array(
        [annotations_2d[f]["centroid"] for f in common_fdi],
        dtype=np.float32,
    )

    # Solves for rotation + uniform scale + translation (similarity transform)
    transform_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if transform_matrix is None:
        raise RuntimeError("Could not compute a transform - check your point data")

    return transform_matrix, common_fdi, src_pts, dst_pts


def visualize_correspondence(photo_path, src_pts, dst_pts, transform_matrix, common_fdi):
    """
    Sanity check: draws where each matched 3D tooth center LANDS after
    applying the computed transform (green), next to where it was ANNOTATED
    in the real photo (red). Close dots = good alignment.
    """
    photo = cv2.imread(photo_path)

    # Apply transform to the 3D centroids to see where they land
    src_pts_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    projected = (transform_matrix @ src_pts_h.T).T

    for i, fdi in enumerate(common_fdi):
        proj_pt = tuple(projected[i].astype(int))
        real_pt = tuple(np.array(dst_pts[i]).astype(int))

        cv2.circle(photo, proj_pt, 6, (0, 255, 0), -1)   # green = projected
        cv2.circle(photo, real_pt, 6, (0, 0, 255), 2)    # red = real annotation
        cv2.putText(photo, str(fdi), real_pt, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    cv2.imshow("Correspondence check (green=projected, red=real) - press any key", photo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Your working 3D case (from Person 2's tooth OBJ data)
    CASE_DIR = r"C:\Users\utkar\Downloads\DENTAL\VITA-3D-Master-Shade-System-for-Dental-Simulators\processed_part1_upper\M7RTPNPC"

    # Look up the 2D patient folder through the index instead of a
    # hardcoded path, since patients are split across 5 separate folders
    PATIENT_ID = "Patient_1"
    patient_index = load_index()  # reads patient_index.json
    if PATIENT_ID not in patient_index:
        raise KeyError(
            f"{PATIENT_ID} not found in patient_index.json - "
            f"run build_patient_index.py again or check the ID"
        )
    PATIENT_DIR = patient_index[PATIENT_ID]

    print(f"Using 3D case: {CASE_DIR}")
    print(f"Using 2D patient: {PATIENT_ID} -> {PATIENT_DIR}")

    centroids_3d, loader = get_3d_tooth_centroids(CASE_DIR)
    photo_path, annotations_2d = load_2d_annotations(PATIENT_DIR)

    # DEBUG: see exactly what FDI numbers (and types) each side has
    print("3D FDI keys:  ", sorted(centroids_3d.keys()), [type(k) for k in list(centroids_3d.keys())[:1]])
    print("2D FDI keys:  ", sorted(annotations_2d.keys()), [type(k) for k in list(annotations_2d.keys())[:1]])

    transform_matrix, common_fdi, src_pts, dst_pts = compute_initial_alignment(
        centroids_3d, annotations_2d
    )

    print("Computed transform matrix:\n", transform_matrix)

    visualize_correspondence(photo_path, src_pts, dst_pts, transform_matrix, common_fdi)