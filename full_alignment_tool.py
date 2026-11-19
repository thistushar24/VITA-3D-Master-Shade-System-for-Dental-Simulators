"""
Phase 3 - COMPLETE pipeline: click a few teeth for an automatic starting
alignment, then fine-tune the result live with sliders (position, rotation,
scale) on top of that starting point.

This is the actual finished version of your phase: auto-fit removes the
blind guessing, sliders fix the small remaining imprecision from manual
clicks - much less trial and error than starting sliders from zero.
"""

import cv2
import numpy as np
from auto_align import get_3d_tooth_centroids, compute_initial_alignment
from generate_overlay import flatten_arch_to_image
from click_align_tool import collect_clicks, MAX_DISPLAY_SIZE


def build_combined_transform(base_matrix, cx, cy, dx, dy, d_angle, d_scale):
    """
    base_matrix: the 2x3 transform from the automatic click-based fit
    cx, cy: center point (in photo pixels) to rotate/scale around - we use
            the base transform's own translation point, which sits roughly
            in the middle of the clicked teeth
    dx, dy: extra pixel offset from the sliders
    d_angle: extra rotation in degrees from the slider
    d_scale: extra scale multiplier from the slider (1.0 = no change)

    Returns the final 2x3 transform combining the auto-fit with the
    slider's manual nudge.
    """
    D = cv2.getRotationMatrix2D((cx, cy), d_angle, d_scale)
    D[0, 2] += dx
    D[1, 2] += dy

    D3 = np.vstack([D, [0, 0, 1]])
    M3 = np.vstack([base_matrix, [0, 0, 1]])
    combined = (D3 @ M3)[:2, :]
    return combined


def run_fine_tune(photo_path, flat_img, min_x, min_y, canvas_scale, padding, base_matrix):
    full_photo = cv2.imread(photo_path)
    ph, pw = full_photo.shape[:2]

    # "canvas pixel -> model space" matrix, reused every frame
    S3 = np.array([
        [1.0 / canvas_scale, 0, min_x - padding / canvas_scale],
        [0, 1.0 / canvas_scale, min_y - padding / canvas_scale],
        [0, 0, 1],
    ])

    cx, cy = base_matrix[0, 2], base_matrix[1, 2]  # rotate/scale around here

    # Build a SMALL working copy of the photo once - all live dragging works
    # on this, not the full-resolution image, so it stays smooth
    preview_scale = min(MAX_DISPLAY_SIZE / pw, MAX_DISPLAY_SIZE / ph, 1.0)
    pw_small, ph_small = int(pw * preview_scale), int(ph * preview_scale)
    small_photo = cv2.resize(full_photo, (pw_small, ph_small))

    # Scales transform OUTPUT directly into small-photo pixel space, so we
    # warp straight into the small canvas instead of warping full-res then
    # shrinking - this is the expensive step we're removing
    P3 = np.array([
        [preview_scale, 0, 0],
        [0, preview_scale, 0],
        [0, 0, 1],
    ])

    window = "Fine-tune: X, Y, Rotation, Scale sliders - 's' to save, 'q' to quit"
    cv2.namedWindow(window)
    cv2.createTrackbar("X offset", window, 200, 400, lambda x: None)      # 200 = 0
    cv2.createTrackbar("Y offset", window, 200, 400, lambda x: None)      # 200 = 0
    cv2.createTrackbar("Rotation", window, 180, 360, lambda x: None)      # 180 = 0 deg
    cv2.createTrackbar("Scale %", window, 100, 300, lambda x: None)       # 100 = 1.0x

    last_final_matrix = None

    while True:
        dx = cv2.getTrackbarPos("X offset", window) - 200
        dy = cv2.getTrackbarPos("Y offset", window) - 200
        d_angle = cv2.getTrackbarPos("Rotation", window) - 180
        d_scale = cv2.getTrackbarPos("Scale %", window) / 100.0

        final_matrix = build_combined_transform(base_matrix, cx, cy, dx, dy, d_angle, d_scale)
        last_final_matrix = final_matrix

        # Combined: canvas pixel -> model -> full photo -> SMALL preview photo
        T_preview = (P3 @ np.vstack([final_matrix, [0, 0, 1]]) @ S3)[:2, :]

        warped_small = cv2.warpAffine(flat_img, T_preview, (pw_small, ph_small))
        alpha = (warped_small[:, :, 3] / 255.0) * 0.55
        canvas = small_photo.astype(np.float32)
        for c in range(3):
            canvas[:, :, c] = alpha * warped_small[:, :, c] + (1 - alpha) * canvas[:, :, c]
        preview = canvas.astype(np.uint8)

        cv2.imshow(window, preview)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            # Only now do we touch the full-resolution photo - once, not per-frame
            T_full = (np.vstack([last_final_matrix, [0, 0, 1]]) @ S3)[:2, :]
            warped_full = cv2.warpAffine(flat_img, T_full, (pw, ph))
            alpha_full = (warped_full[:, :, 3] / 255.0) * 0.55
            full_canvas = full_photo.astype(np.float32)
            for c in range(3):
                full_canvas[:, :, c] = alpha_full * warped_full[:, :, c] + (1 - alpha_full) * full_canvas[:, :, c]
            cv2.imwrite("final_fine_tuned_result.png", full_canvas.astype(np.uint8))
            print("Saved final_fine_tuned_result.png (full resolution)")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    CASE_DIR = r"C:\Users\utkar\Downloads\DENTAL\VITA-3D-Master-Shade-System-for-Dental-Simulators\processed_part1_upper\M7RTPNPC"
    PHOTO_PATH = "patient_photo.jpg"

    print("Step 1: loading your 3D model...")
    centroids_3d, loader = get_3d_tooth_centroids(CASE_DIR)

    print("\nStep 2: click each tooth shown in the on-screen instruction, in order.")
    print("Press 'n' to skip a tooth you can't see clearly. Press 'q' when done.\n")
    clicked = collect_clicks(PHOTO_PATH)

    if len(clicked) < 3:
        raise ValueError(f"Only {len(clicked)} teeth clicked - need at least 3.")

    annotations_2d = {fdi: {"centroid": pt} for fdi, pt in clicked.items()}

    print("\nStep 3: computing the automatic starting alignment...")
    base_matrix, common_fdi, src_pts, dst_pts = compute_initial_alignment(centroids_3d, annotations_2d)
    print("Base transform:\n", base_matrix)

    print("\nStep 4: opening the fine-tune window. Nudge sliders to perfect the fit.")
    print("Press 's' to save your final result, 'q' to quit.\n")
    flat_img, min_x, min_y, canvas_scale, padding = flatten_arch_to_image(loader)
    run_fine_tune(PHOTO_PATH, flat_img, min_x, min_y, canvas_scale, padding, base_matrix)