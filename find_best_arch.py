"""
find_best_arch.py  -  pick the 3D arch that best matches a photo.

The 3D scans and the photo are different people, so the arch you happen to
load first is rarely the best fit.  Your library has ~150 upper arches; this
ranks them by how well their tooth layout matches the teeth in the photo.

First run builds a small cache (arch_library_cache.json): it reads every
case once (a few minutes for all 150; use --limit to do a part, the cache
keeps growing on later runs).  After that, searching takes about a second.

Use ONE of:
    python find_best_arch.py --clicks final_pose.json   (your own photo, saved by dental_overlay.py)
    python find_best_arch.py --patient Patient_1         (a dataset photo, no clicks)
"""
import argparse
import concurrent.futures as cf
import json
import os
import time

import numpy as np

import dental_overlay as do

CACHE_FILE = "arch_library_cache.json"
RANK_MAX_INDEX = 6     # teeth used for ranking (1..6)
CACHE_VER = 2          # bump when the cached layout format changes (old entries are re-indexed)


def list_cases(library):
    out = []
    for name in sorted(os.listdir(library)):
        p = os.path.join(library, name)
        if os.path.isdir(p):
            try:
                if any(f.startswith("tooth_") and f.endswith(".obj") for f in os.listdir(p)):
                    out.append(p)
            except OSError:
                pass
    return out


def index_case(case_dir):
    """Reference points of one arch (cheap: only what the ranking needs)."""
    try:
        teeth = do.prepare_teeth(do.load_case(case_dir, target_faces=250,
                                              max_index=6, quiet=True))
        frame = do.anatomical_frame(teeth, quiet=True)
        cam = do.to_camera_frame(teeth, frame, False, frame["auto_flip_v"])
        return case_dir, visible_layout(cam), None
    except Exception as exc:  # noqa
        return case_dir, None, str(exc)


def visible_layout(cam, px_per_mm=20.0):
    """Where each tooth's VISIBLE part is centred in a standard front view
    (back teeth partly hidden), in image-plane mm.  This is what a person
    clicks, so it compares like with like."""
    ren = do.ArchRenderer(cam)
    fid = np.unique(ren.F)
    z, _ = do.plane_coords(ren.V[fid], do.DEFAULT_PERSP / 1e4, 0, 0)
    x0, x1, y0, y1 = z.real.min(), z.real.max(), z.imag.min(), z.imag.max()
    s, pad = px_per_mm, 30
    W, H = int((x1 - x0) * s + 2 * pad), int((y1 - y0) * s + 2 * pad)
    a0 = complex(s, 0)
    b0 = complex(pad - s * x0, pad - s * y0)
    rend = ren.render(do.default_pose_dict(a0, b0), W, H, 1.0)
    lab = rend["lab"]
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs].astype(np.int64)
    n = len(ren.fdis) + 1
    cnt = np.bincount(ids, minlength=n).astype(float)
    cx = np.bincount(ids, weights=xs, minlength=n) / np.maximum(cnt, 1)
    cy = np.bincount(ids, weights=ys, minlength=n) / np.maximum(cnt, 1)
    out = {}
    for f in ren.fdis:
        i = ren.label_id[f]
        if cnt[i] >= 20:
            p = (complex(cx[i], cy[i]) - b0) / a0           # visible centre in image-plane mm
            w = float(cam[f]["ref"][2])                       # its depth (behind the incisors)
            k = 1.0 / max(0.25, 1.0 + (do.DEFAULT_PERSP / 1e4) * w)
            out[str(f)] = [float(p.real / k), float(p.imag / k), w]   # undo perspective -> 3D
    return out


def load_cache(path):
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return {}


def build_cache(cases, cache, limit, workers):
    todo = [c for c in cases if c not in cache or cache[c].get("ver") != CACHE_VER][:limit]
    if not todo:
        return cache
    print(f"Indexing {len(todo)} arches ({len(cache)} already cached) ...")
    t0, done = time.time(), 0

    def absorb(res):
        nonlocal done
        case, refs, err = res
        cache[case] = {"refs": refs, "ver": CACHE_VER} if refs else {"error": err, "ver": CACHE_VER}
        done += 1
        if done % 5 == 0 or done == len(todo):
            el = time.time() - t0
            print(f"  {done}/{len(todo)}  ({el:.0f}s, ~{el / done * (len(todo) - done):.0f}s left)")
            with open(CACHE_FILE, "w") as fh:
                json.dump(cache, fh)

    if workers > 1:
        try:
            with cf.ProcessPoolExecutor(max_workers=workers) as ex:
                for res in ex.map(index_case, todo):
                    absorb(res)
            return cache
        except Exception as exc:  # noqa  (fall back to one process)
            print(f"  parallel indexing failed ({exc}); continuing one at a time")
            todo = [c for c in todo if c not in cache]
    for c in todo:
        absorb(index_case(c))
    return cache


def rank(cache, clicks):
    rows = []
    for case, entry in cache.items():
        lay = entry.get("refs")                 # visible-centre layout per tooth (u, v, w)
        if not lay or entry.get("ver") != CACHE_VER:
            continue
        cam = {int(f): {"ref": np.array(p)} for f, p in lay.items()}
        common = [f for f in clicks if f in cam and f % 10 <= RANK_MAX_INDEX]
        if len(common) < 4:
            continue
        try:                                     # same camera fit the overlay uses
            pose = do.fit_pose(cam, {f: clicks[f] for f in common}, extended=True)
        except Exception:  # noqa
            continue
        rows.append((pose["rms_mm"], case, len(common)))
    rows.sort()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--library", default=os.path.dirname(do.DEFAULT_CASE),
                    help="folder that contains the (upper) case folders")
    ap.add_argument("--library-lower", default=None, help="folder that contains the lower case folders")
    ap.add_argument("--both", action="store_true",
                    help="also pick the best LOWER arch (library found next to the upper one)")
    ap.add_argument("--clicks", default=None, help="final_pose.json from dental_overlay.py")
    ap.add_argument("--patient", default=None, help="dataset patient folder or ID")
    ap.add_argument("--limit", type=int, default=200, help="max new arches to index this run")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if args.clicks:
        clicks = do.load_clicks_file(args.clicks)
        how = f'--clicks "{args.clicks}"'
    elif args.patient:
        _, cents, _ = do.load_patient(args.patient)
        clicks = {f: p for f, p in cents.items() if f % 10 <= do.MAX_CLICK_INDEX}
        how = f"--patient {args.patient}"
    else:
        raise SystemExit("Give --clicks final_pose.json  or  --patient Patient_1")

    libs = [("upper", args.library)]
    lower_lib = args.library_lower or (do.derive_lower(args.library) if args.both else None)
    if args.both and not lower_lib:
        print("Could not find the lower library next to the upper one - use --library-lower PATH.")
    if lower_lib:
        libs.append(("lower", lower_lib))

    picks = {}
    cache = load_cache(CACHE_FILE)
    for jaw, library in libs:
        cases = list_cases(library)
        if not cases:
            print(f"No case folders found in {library}")
            continue
        print(f"\n{len(cases)} arches in {library}")
        cache = build_cache(cases, cache, args.limit, args.workers)
        with open(CACHE_FILE, "w") as fh:
            json.dump(cache, fh)
        mine = {c: cache[c] for c in cases if c in cache}
        rows = rank(mine, clicks)
        if not rows:
            print(f"No indexed arch in this library had at least 4 teeth in common with the photo "
                  f"({jaw} jaw: did you click those teeth?).")
            continue
        print(f"Best-fitting arches ({jaw}; residual in mm, lower = better; {len(rows)} compared):")
        for i, (rms, case, n) in enumerate(rows[:args.top], 1):
            print(f"  {i}. {rms:.2f} mm   {os.path.basename(case)}   ({n} teeth)")
        picks[jaw] = rows[0][1]
    if not picks:
        raise SystemExit("Nothing could be ranked.")

    photo_arg = "" if args.patient else " --photo patient_photo.jpg"
    if "upper" in picks:
        cmd = f'python dental_overlay.py --case "{picks["upper"]}"'
        if "lower" in picks:
            cmd += f' --case-lower "{picks["lower"]}"'
    else:
        cmd = f'python dental_overlay.py --case "{picks["lower"]}"'
    print("\nRun the overlay with the best one(s):")
    print(f"  {cmd} {how}{photo_arg}")


if __name__ == "__main__":
    main()