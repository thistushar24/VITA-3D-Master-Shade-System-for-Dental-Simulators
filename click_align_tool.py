"""
Phase 3 - Click-based alignment (simplified - no typing mid-process).

Click teeth in a fixed order shown on screen. The displayed image is
scaled down to fit your screen if it's too large (common with phone
photos), while all alignment math still uses the REAL full-resolution
pixel coordinates underneath.
"""

import cv2
import numpy as np
from auto_align import get_3d_tooth_centroids, compute_initial_alignment
from generate_overlay import flatten_arch_to_image, composite_overlay

# Click teeth in exactly this order (upper arch, left-to-right as seen in
# a typical frontal photo). Skip any tooth you genuinely can't see by
# pressing 'n' to move to the next one in the list without clicking.
CLICK_ORDER = [15, 14, 13, 12, 11, 21, 22, 23, 24, 25]

MAX_DISPLAY_SIZE = 900  # max width/height shown on screen, in pixels

clicked_points = []        # [(x, y, fdi), ...] - in REAL photo coordinates
current_index = [0]
display_img = None
scale_factor = [1.0]       # real_pixel = displayed_pixel / scale_factor


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and current_index[0] < len(CLICK_ORDER):
        # Convert the click (in the resized display) back to real photo pixels
        real_x = int(x / scale_factor[0])
        real_y = int(y / scale_factor[0])
        fdi = CLICK_ORDER[current_index[0]]
        clicked_points.append((real_x, real_y, fdi))
        print(f"Recorded tooth {fdi} at real photo coords ({real_x}, {real_y})")
        current_index[0] += 1


def collect_clicks(photo_path):
    global display_img

    full_img = cv2.imread(photo_path)
    if full_img is None:
        raise FileNotFoundError(f"Could not load {photo_path}")

    h, w = full_img.shape[:2]
    scale = min(MAX_DISPLAY_SIZE / w, MAX_DISPLAY_SIZE / h, 1.0)  # never upscale
    scale_factor[0] = scale

    if scale < 1.0:
        print(f"Photo is {w}x{h} - too large for the screen, showing it scaled "
              f"down to {int(w*scale)}x{int(h*scale)} for clicking "
              f"(alignment still uses full resolution).")

    display_img = cv2.resize(full_img, (int(w * scale), int(h * scale)))

    window = "Click teeth in order shown - 'n' to skip, 'q' to finish"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    while True:
        frame = display_img.copy()

        for (rx, ry, fdi) in clicked_points:
            # Convert real coords back to display coords just for drawing
            dx, dy = int(rx * scale_factor[0]), int(ry * scale_factor[0])
            cv2.circle(frame, (dx, dy), 6, (0, 0, 255), -1)
            cv2.putText(frame, str(fdi), (dx + 8, dy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        if current_index[0] < len(CLICK_ORDER):
            next_fdi = CLICK_ORDER[current_index[0]]
            instruction = f"Click tooth FDI {next_fdi}  (or press 'n' to skip)"
        else:
            instruction = "All teeth done - press 'q' to continue"

        cv2.putText(frame, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.imshow(window, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n") and current_index[0] < len(CLICK_ORDER):
            skipped = CLICK_ORDER[current_index[0]]
            print(f"Skipped tooth {skipped}")
            current_index[0] += 1

    cv2.destroyAllWindows()
    return {fdi: (x, y) for (x, y, fdi) in clicked_points}


if __name__ == "__main__":
    CASE_DIR = r"C:\Users\utkar\Downloads\DENTAL\VITA-3D-Master-Shade-System-for-Dental-Simulators\processed_part1_upper\M7RTPNPC"
    PHOTO_PATH = "patient_photo.jpg"

    print("Step 1: loading your 3D model...")
    centroids_3d, loader = get_3d_tooth_centroids(CASE_DIR)
    print("Teeth available in this 3D model:", sorted(centroids_3d.keys()))

    print("\nStep 2: click each tooth shown in the on-screen instruction, in order.")
    print("Press 'n' to skip a tooth you can't see clearly. Press 'q' when done.\n")
    clicked = collect_clicks(PHOTO_PATH)

    if len(clicked) < 3:
        raise ValueError(
            f"Only {len(clicked)} teeth were clicked - need at least 3. "
            f"Run it again and click more teeth before pressing 'q'."
        )

    annotations_2d = {fdi: {"centroid": pt} for fdi, pt in clicked.items()}

    print("\nStep 3: computing alignment from your clicks...")
    transform_matrix, common_fdi, src_pts, dst_pts = compute_initial_alignment(
        centroids_3d, annotations_2d
    )
    print("Used teeth:", common_fdi)
    print("Computed transform:\n", transform_matrix)

    print("\nStep 4: rendering the overlay...")
    flat_img, min_x, min_y, canvas_scale, padding = flatten_arch_to_image(loader)
    result = composite_overlay(PHOTO_PATH, flat_img, min_x, min_y, canvas_scale, padding, transform_matrix)

    # Save the FULL resolution result to disk first
    cv2.imwrite("click_align_result.png", result)
    print("Saved to click_align_result.png")

    # Show a SCALED-DOWN preview only - same large-image fix as the click step
    rh, rw = result.shape[:2]
    preview_scale = min(MAX_DISPLAY_SIZE / rw, MAX_DISPLAY_SIZE / rh, 1.0)
    preview = cv2.resize(result, (int(rw * preview_scale), int(rh * preview_scale)))

    cv2.imshow("Final overlay preview - press any key to close", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()