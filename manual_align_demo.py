"""
Phase 3 - Step 2: Manual 2D-3D overlay alignment (DEMO / prototype version)

Renders the 3D arch to a transparent PNG, then overlays it on a patient
photo. Sliders let you translate / rotate / scale the overlay until it
visually lines up with the teeth in the photo.

This is the "quick demo" path: no camera calibration, no markers required
yet - just a 2D affine overlay. Good enough to prove the concept before
building the full perspective-correct version.
"""

import json
import cv2
import numpy as np
import open3d as o3d
from arch_loader import ArchLoader


def render_arch_to_image(case_dir, image_size=(800, 600)):
    """
    Renders the 3D arch from a fixed frontal angle to an RGBA image
    with a transparent background, using Open3D's offscreen renderer.
    """
    loader = ArchLoader(case_dir)
    arch = loader.get_combined_arch()

    renderer = o3d.visualization.rendering.OffscreenRenderer(*image_size)
    renderer.scene.set_background([0, 0, 0, 0])  # transparent

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.add_geometry("arch", arch, material)

    # Frontal camera looking straight at the arch center
    center = arch.get_center()
    bbox = arch.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_extent())
    eye = center + [0, 0, extent * 1.5]
    renderer.setup_camera(60.0, center, eye, [0, 1, 0])

    img = renderer.render_to_image()
    o3d.io.write_image("arch_render.png", img)
    return np.asarray(img)


def overlay_alignment_tool(arch_rgba_path, photo_path):
    """
    Interactive overlay tool.
    Sliders: X, Y position | Rotation | Scale | Opacity
    Press 's' to save the current transform to alignment_transform.json,
    'q' to quit.
    """
    arch_img = cv2.imread(arch_rgba_path, cv2.IMREAD_UNCHANGED)  # RGBA
    photo = cv2.imread(photo_path)
    if arch_img is None or photo is None:
        raise FileNotFoundError("Check arch_rgba_path / photo_path")

    ph, pw = photo.shape[:2]

    window = "Manual Alignment - press 's' to save, 'q' to quit"
    cv2.namedWindow(window)
    cv2.createTrackbar("X", window, pw // 2, pw, lambda x: None)
    cv2.createTrackbar("Y", window, ph // 2, ph, lambda x: None)
    cv2.createTrackbar("Rotation", window, 180, 360, lambda x: None)  # 180 = 0 deg
    cv2.createTrackbar("Scale %", window, 100, 300, lambda x: None)
    cv2.createTrackbar("Opacity %", window, 70, 100, lambda x: None)

    while True:
        x = cv2.getTrackbarPos("X", window)
        y = cv2.getTrackbarPos("Y", window)
        angle = cv2.getTrackbarPos("Rotation", window) - 180
        scale = cv2.getTrackbarPos("Scale %", window) / 100.0
        opacity = cv2.getTrackbarPos("Opacity %", window) / 100.0

        ah, aw = arch_img.shape[:2]
        M = cv2.getRotationMatrix2D((aw / 2, ah / 2), angle, scale)
        rotated = cv2.warpAffine(
            arch_img, M, (aw, ah),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        canvas = photo.copy()
        top_left_x = x - aw // 2
        top_left_y = y - ah // 2

        for c in range(3):
            x1, x2 = max(top_left_x, 0), min(top_left_x + aw, pw)
            y1, y2 = max(top_left_y, 0), min(top_left_y + ah, ph)
            ax1, ay1 = x1 - top_left_x, y1 - top_left_y
            ax2, ay2 = ax1 + (x2 - x1), ay1 + (y2 - y1)

            if x2 <= x1 or y2 <= y1:
                continue

            alpha = (rotated[ay1:ay2, ax1:ax2, 3] / 255.0) * opacity
            canvas[y1:y2, x1:x2, c] = (
                alpha * rotated[ay1:ay2, ax1:ax2, c]
                + (1 - alpha) * canvas[y1:y2, x1:x2, c]
            )

        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            transform = {"x": x, "y": y, "rotation_deg": angle, "scale": scale}
            print("Saved transform:", transform)
            with open("alignment_transform.json", "w") as f:
                json.dump(transform, f, indent=2)
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    CASE_DIR = "processed_part1_upper/BUGB9DY1"

    # Step 1: render the arch to a transparent PNG (run once per case)
    render_arch_to_image(CASE_DIR)

    # Step 2: overlay it on a patient photo and drag it into place.
    # Swap in any frontal teeth photo for now while you wait on the
    # photo-processing pipeline's real marker/segmentation output.
    overlay_alignment_tool("arch_render.png", "patient_photo.jpg")