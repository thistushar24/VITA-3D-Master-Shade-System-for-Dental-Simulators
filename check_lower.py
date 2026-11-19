"""
check_lower.py  -  shows exactly where your LOWER-jaw tooth files are and prints the command to run.
Needs nothing but Python.   Run:   python check_lower.py
(If your repo is somewhere else:   python check_lower.py --root "C:\\path\\to\\VITA-3D-...-Simulators")
"""
import argparse
import glob
import os

ROOT = r"C:\Users\utkar\Downloads\DENTAL\VITA-3D-Master-Shade-System-for-Dental-Simulators"


def describe(folder):
    objs = sorted(glob.glob(os.path.join(folder, "tooth_*.obj")))
    ids = sorted(int(os.path.basename(o)[6:-4]) for o in objs if os.path.basename(o)[6:-4].isdigit())
    small = [o for o in objs if os.path.getsize(o) < 2048]
    return objs, ids, small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--upper-case", default="M7RTPNPC")
    args = ap.parse_args()
    up_lib = os.path.join(args.root, "processed_part1_upper")
    lo_lib = os.path.join(args.root, "processed_part1_lower")
    print(f"Repo folder     : {args.root}  -> {'found' if os.path.isdir(args.root) else 'NOT FOUND'}")
    print(f"Upper library   : {'found' if os.path.isdir(up_lib) else 'NOT FOUND'}")
    print(f"Lower library   : {'found' if os.path.isdir(lo_lib) else 'NOT FOUND'}")
    if not os.path.isdir(lo_lib):
        print("\n=> The lower-jaw folder is missing from your copy of the repo.")
        print("   Open the repo in File Explorer and look for processed_part1_lower.")
        print("   If it is not there, pull it (git pull, then git lfs pull) or ask Tushar for that folder.")
        return
    cases = sorted(d for d in os.listdir(lo_lib) if os.path.isdir(os.path.join(lo_lib, d)))
    print(f"Lower cases     : {len(cases)} folders")
    same = os.path.join(lo_lib, args.upper_case)
    print(f"Same ID as your upper case ({args.upper_case}): {'YES' if os.path.isdir(same) else 'no (that is fine)'}")
    good = None
    for c in cases:
        objs, ids, small = describe(os.path.join(lo_lib, c))
        if len(objs) >= 8 and not small:
            good = (c, objs, ids)
            break
    if good is None:
        print("\n=> No lower case with at least 8 real tooth files was found.")
        for c in cases[:5]:
            objs, ids, small = describe(os.path.join(lo_lib, c))
            print(f"   {c}: {len(objs)} tooth files, {len(small)} look like Git-LFS stubs (tiny files)")
        print("   If files are tiny stubs, run:  git lfs pull   inside the repo folder.")
        return
    c, objs, ids = good
    folder = os.path.join(lo_lib, c)
    print(f"\nA usable lower case : {c}  ({len(objs)} teeth: {ids})")
    print(f"  folder: {folder}")
    print(f"  first file size: {os.path.getsize(objs[0]):,} bytes (real data, not a stub)")
    up_case = os.path.join(up_lib, args.upper_case)
    print("\nCOPY ONE OF THESE:\n")
    print("1) Lower jaw ALONE (quickest test that it works):")
    print(f'   python dental_overlay.py --case "{folder}" --photo patient_photo.jpg\n')
    print("2) Upper + lower together, your own photo:")
    print(f'   python dental_overlay.py --case "{up_case}" --case-lower "{folder}" --photo patient_photo.jpg\n')
    print("3) Upper + lower together, a dataset photo (no clicking):")
    print(f'   python dental_overlay.py --case "{up_case}" --case-lower "{folder}" --patient Patient_1')


if __name__ == "__main__":
    main()