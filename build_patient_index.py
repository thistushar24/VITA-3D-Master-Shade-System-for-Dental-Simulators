"""
Google Drive splits large folder downloads into multiple zip parts
(e.g. data-xxx-1-001, -002, -003...). Rather than manually merging their
contents into one folder, this scans all of them once and builds an index
of patient_id -> actual folder path, so the rest of your code can look up
any patient without caring which split folder it physically lives in.
"""

import os
import glob
import json


def build_patient_index(split_folders):
    """
    split_folders: list of paths to the extracted split folders, e.g.
        [r"C:\\...\\data-20261002T034339Z-1-001",
         r"C:\\...\\data-20261002T034339Z-1-002", ...]

    Returns: {patient_id (str): full_path_to_that_patient_folder}
    """
    index = {}

    for folder in split_folders:
        # Find every Patient_X folder inside this split, at any depth
        # (in case Drive nested an extra folder level during zip/extract)
        pattern = os.path.join(folder, "**", "Patient_*")
        matches = glob.glob(pattern, recursive=True)

        for match in matches:
            if os.path.isdir(match):
                patient_id = os.path.basename(match)
                index[patient_id] = match

    print(f"Indexed {len(index)} patients across {len(split_folders)} folders")
    return index


def save_index(index, out_path="patient_index.json"):
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved index to {out_path}")


def load_index(path="patient_index.json"):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # List every split folder you extracted from the Drive zips
    SPLIT_FOLDERS = [
        r"C:\Users\utkar\Downloads\data-20261002T034339Z-1-001",
        r"C:\Users\utkar\Downloads\data-20261002T034339Z-1-002",
        r"C:\Users\utkar\Downloads\data-20261002T034339Z-1-003",
        r"C:\Users\utkar\Downloads\data-20261002T034339Z-1-004",
        r"C:\Users\utkar\Downloads\data-20261002T034339Z-1-005",
    ]

    index = build_patient_index(SPLIT_FOLDERS)
    save_index(index)

    # Quick sanity check - look up a couple of patients
    for pid in list(index.keys())[:3]:
        print(pid, "->", index[pid])