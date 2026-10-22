# v1 — automated 2D-to-3D smile simulation demo

Journey: upload Photo A (frontal, retractor) + Photo B (smiling face) → automatic best-fit model → align over Photo A → confirm →
edit the 3D working copy (versions) → place over Photo B (own placement) → before/after → export PNG/JPEG.

This is a research prototype. The 3D model is a template from a library; results are simulations, not clinical predictions.

## Run locally
```bash
# from repo root, once
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
(cd versions/v1 && npm ci && npm run build)

# single process serves API + UI at http://localhost:8000
cd backend && ../.venv/bin/uvicorn dental_reg.app:app --port 8000

# or UI dev server with hot reload (second terminal): http://localhost:5174, proxies /api to :8000
cd versions/v1 && npm run dev
```
Press **Load synthetic demo photos** to try it without patient images. (Demo canine points are pre-placed on the model's true
canines, standing in for the user dragging them.) Tests: `cd backend && ../.venv/bin/python -m pytest -q`.

## What this folder contains / reuses
`versions/v1/` has only the demo entry point (`index.html`, `src/App.jsx`, `src/api.js`, Vite config). It reuses the shared
viewer/maths in `frontend/src` (alias `@shared`) and the backend in `backend/dental_reg` (`vision.py`, `match.py`, `library.py`,
`v1.py`). The model library is `data/demo_library/` (shared, not copied). The earlier `frontend/` app (registration workbench) is untouched
apart from moving the mesh-drawing function to `frontend/src/draw.js`.

## Model catalogue (`data/demo_library/catalogue.json`, built by `backend/build_demo_library.py`)
Per model: id, file, jaw, classification (`complete_arch` = ≥12 teeth incl. both canines and central incisors; `partial_arch`;
`single_mesh`), tooth labels/count, bbox/extent (mm), canine distance, incisor widths, shade (`null`: the source repository has no shade
data), source, licence note, preprocessing status. The published library has **22 procedurally generated, non-anatomical arches**
(20 upper, 2 lower) so the demo, tests and deployment need no third-party data. **Real Teeth3DS models are CC BY-NC-ND (no derivatives) and
are not committed** - see `docs/ASSETS_AND_LICENSES.md`. Plug in a private real library with `DENTAL_LIBRARY_DIR` (build it with
`python backend/build_library.py OUT_DIR CASES_ROOT...` from your own licensed copy; needs the real OBJ files, `git lfs pull`).

## How automatic best-fit selection works
1. **Tooth region** (`vision.segment_teeth`): bright/low-red seed (central incisors) grown with GrabCut in a window; upper arch =
   top 75% of the region (heuristic). Canine points = estimated from the incisor-pair width, **user-draggable**. Heuristic, colour-based,
   not a trained model; it can fail on unusual lighting/colour (then draw the outline manually).
2. **Stage 1, all upper candidates (cheap):** `0.6·shapeIoU + 0.4·canineRatioScore`. shapeIoU = IoU of the model's frontal silhouette
   and the photo mask, both cropped to their bounding boxes and resized to 96×48 (compares aspect ratio and tooth arrangement, not
   absolute size). canineRatioScore compares canine-distance/width ratios.
3. **Stage 2, top 6:** full registration (PnP/silhouette fit with the two canines as landmark residuals), then
   `0.5·fitIoU + 0.3·canineFit + 0.2·stage1`, `canineFit = 1 − rmse_px/(0.08·mask_width)`.
4. Partial arches rank after complete ones. The top candidate is selected automatically. It is flagged **preliminary** if its final score
   is < 0.7, it is within 0.02 of the runner-up, or it is not a complete arch. Weights/thresholds are hand-set, not learned.
   Reported numbers are agreement measures with the photo, **not accuracy or probability**. Deterministic (fixed seeds): same input →
   same ranking (tested). Shade is never used.
5. The user can replace the model from the catalogue at any time.

### Measured results (and one correction)
- **Bug found and fixed in this round:** the silhouette rasteriser used `cv2.fillPoly`, whose even-odd rule made overlapping triangles
  cancel, so every earlier IoU (including the "≈0.6 on a real photo" I first reported) was too low and the optimiser was fitting the wrong
  target. Silhouettes are now per-tooth convex hulls via `fillConvexPoly` (covered by tests). All numbers below are after the fix.
- **Real photo (1 retractor photo, private library of 36 real-derived arches, local only):** automatic segmentation + ranking took ~5 s;
  selected `upper_E4GH1D3S`, fitted silhouette IoU 0.73, canine reprojection error 30 px, flagged *preliminary* (runner-up within 0.003).
  Visual check: arch width and tooth positions follow the photo; the template extends onto the gingiva. One photo is an anecdote, not an evaluation.
- **Canine estimate** (`backend/bench_canines.py`): estimated from the upper-tooth mask (midline = bbox centre, offset = 0.33 × width, where
  0.33 is the measured median canine-distance/arch-width of 36 real arches). On synthetic renders of those 36 arches: median error 10.5 %
  of the canine distance, p90 17.9 %, worst 63 % (n=72). In-sample for the 0.33 constant and synthetic only. On the real photo the points land
  within ~15-35 px of canines placed by eye. Points stay draggable; the old incisor-width estimate (median 16 %, p90 50 %, and a total miss on
  the synthetic demo) was replaced.

## Limits (v1)
- Only the upper arch is matched (a lower-arch library of real scans is not available/licensed here); the 2 procedural lower models are not used for matching.
- Segmentation/canine estimate are heuristics tested on one real photo and synthetic images.
- Photo A and B use an approximate (uncalibrated) camera; two canine points do not fix depth/pose.
- "Visible-region mask" is hand-drawn polygons (no automatic lip detection). Layer order = model-on-top (polygons hide model) or
  photo-on-top (model shows only inside polygons).
- Sessions, registrations and uploads are written under `DENTAL_STORAGE` and reloaded after a restart (tested). On Render's free tier that disk is ephemeral, so a redeploy still wipes it; attach a Render disk to keep it. Photo B export is client side.
- No shade-selection feature exists in the repository (Person 2's repo has no such code), so none was preserved or integrated.

## Deploy on Render (single Docker service)
`render.yaml` + `Dockerfile` at the repo root. In the dashboard: **New → Blueprint** → select the repo/branch → apply (or New Web
Service, Runtime *Docker*, health check path `/api/health`). No secrets are needed; see `.env.example`. Render injects `PORT`; the
container binds `0.0.0.0:$PORT`. UI and API share one origin, so no CORS or API URL setup is required. The Docker build was **not
executed** during development (no Docker daemon); equivalent steps (clean `npm ci && vite build`, fresh pinned venv, server started
from the copied layout) were run instead.

## Deploy on Vercel (two projects from this repo)

Vercel runs Python as short-lived serverless functions, so the demo runs there in **stateless mode**: the browser keeps the session
(photos, annotations, versions, placements) and the API only computes (`/api/v1/stateless/{analyse,match,place,transform}`, plus the
read-only `/api/v1/catalogue`, `/api/models/lib/<id>/mesh`, `/api/samples/*`, `/api/health`). No database or object storage is needed and no
paid service is involved. The stateful API (`/api/v1/sessions…`, server-side sessions) is unchanged and remains the mode for local/Docker/Render.

| | API project | UI project |
|---|---|---|
| Root Directory | repo root (`.`) | `versions/v1` |
| Framework | FastAPI/Python (auto; `app.py` exposes `app`, `requirements.txt`, `.python-version` = 3.12) | Vite (auto; `versions/v1/vercel.json`) |
| Build | none (dependencies only) | `npm ci && npm run build` → `dist` |
| Env vars | `DENTAL_CORS=https://<ui>.vercel.app` (and optionally `DENTAL_CORS_REGEX=https://<project>-.*\.vercel\.app` for previews) | `VITE_API_BASE=https://<api>.vercel.app` (no trailing slash). `VITE_API_MODE=stateless` is set by `vercel.json` |
| Other setting | none | keep **"Include source files outside of the Root Directory in the Build Step"** enabled (shared code is in `frontend/src`) |

Steps: (1) import the repo, branch `release/v1-vercel`, Root Directory `.` → deploy → note the URL, open `<url>/api/health`;
(2) import the same repo again, Root Directory `versions/v1`, set `VITE_API_BASE` to the API URL → deploy;
(3) set `DENTAL_CORS` on the API project to the UI URL and redeploy the API. Both projects can use the free Hobby plan for a demo.

Limits that shaped the design (from Vercel docs, checked 2026-10): request/response body **4.5 MB** (so Photo A is downscaled in the browser
to ≤1600 px before upload; Photo B never leaves the browser; the API rejects >4 MB with a clear error); function filesystem is read-only
except per-instance `/tmp` (the API writes nothing it needs later in stateless mode); Python bundle limit 500 MB (a local `vercel` CLI build with Linux wheels measured **325 MB**, above the 250 MB
"standard" size, so Vercel switches to *runtime dependency installation* - expect slower cold starts; not yet confirmed on the platform since no
deployment was possible); 2 GB memory and 300 s default duration (a match takes ≈3-6 s; first request after idle also pays the
import/cold-start cost).
Consequences: closing the tab loses the session (the UI offers PNG/JPEG export; there is no server-side save); the 20 procedural arches
are the only models (licence-clean). To add durable server-side sessions on Vercel you would need an external store (e.g. Vercel Blob for
photos + a KV/Postgres for session JSON; both have free tiers but need an account and configuration) - not implemented.

**Status of this deployment path:** configuration validated against Vercel's published `vercel.json` schema; the API runs locally with `VERCEL=1` from the repo
root exactly as Vercel would start it, and the stateless UI was tested against it cross-origin (see the test notes in the project hand-off). **It has not been
deployed to Vercel**: this environment has no Vercel login or token (`vercel whoami` → logged out) and an anonymous `vercel deploy --temporary` was rejected with
"Not authorized". Run the steps above from an authenticated account.
