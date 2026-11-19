"""
Loads the 2D intraoral photo annotations (tooth_contours.json) for one patient.

Expected structure per the dataset README:
    Patient_X/
        original.png
        tooth_contours.json
"""

import json
import os


def load_2d_annotations(patient_dir):
    """
    Returns:
        photo_path: full path to original.png
        annotations: {fdi_number: {"centroid": (x, y), "contour": [[x,y], ...]}}
    """
    json_path = os.path.join(patient_dir, "tooth_contours.json")
    photo_path = os.path.join(patient_dir, "original.png")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No tooth_contours.json in {patient_dir}")
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"No original.png in {patient_dir}")

    with open(json_path, "r") as f:
        data = json.load(f)

    annotations = {}
    for tooth in data["teeth"]:
        fdi = int(tooth["fdi"])  # normalize to int - JSON may store as string
        centroid = (tooth["centroid"]["x"], tooth["centroid"]["y"])
        contour = tooth["contour"]
        annotations[fdi] = {"centroid": centroid, "contour": contour}

    print(f"Loaded {len(annotations)} annotated teeth for {data.get('patient_id')}")
    return photo_path, annotations


if __name__ == "__main__":
    PATIENT_DIR = "Patient_1000"
    photo_path, annotations = load_2d_annotations(PATIENT_DIR)
    print("Photo:", photo_path)
    print("FDI numbers found:", sorted(annotations.keys()))