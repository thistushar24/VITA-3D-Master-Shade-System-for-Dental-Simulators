"""Build data/library from folders of per-tooth OBJ cases (Person 2 format).

usage: python build_library.py OUT_DIR CASES_ROOT [CASES_ROOT ...] [--faces 3500]
Each CASES_ROOT contains <case_id>/tooth_<FDI>.obj. Originals are only read.
"""
import json
import sys
from pathlib import Path

from dental_reg import core, library


def main(argv):
    faces = 3500
    if "--faces" in argv:
        i = argv.index("--faces")
        faces = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    out, roots = Path(argv[0]), [Path(r) for r in argv[1:]]
    out.mkdir(parents=True, exist_ok=True)
    models = []
    for root in roots:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            try:
                full = core.load_arch_dir(d)
            except core.InputError as e:
                print("skip", d.name, e)
                continue
            small = core.decimate(full, faces)
            mid = f"{full.jaw}_{d.name}"
            library.save_arch_npz(small, out / f"{mid}.npz", centroids=full.teeth)
            small.teeth = full.teeth
            models.append(library.build_entry(small, mid, f"{mid}.npz"))
            print("added", mid, models[-1]["classification"], len(small.faces), "faces")
    (out / "catalogue.json").write_text(json.dumps({"schema": "dental-library/1", "models": models}, indent=1))
    print(len(models), "models ->", out)


if __name__ == "__main__":
    main(sys.argv[1:])
