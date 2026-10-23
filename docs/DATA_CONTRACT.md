# Data contract, JSON schema and API

## Coordinate conventions
- **Image**: pixels, origin top-left, x right, y down. Photo PNG/JPEG; masks 8-bit PNG (>127 = tooth) or polygons in JSON.
- **Model**: millimetres, Teeth3DS scanner frame, files never modified. Per-tooth OBJ named `tooth_<FDI>.obj` (FDI 11–48);
  teeth are in a shared arch frame. In this frame −y is labial (incisors), +x is the patient's right (FDI 1x/4x), −z is "up" (verified visually).
- **Camera**: OpenCV (+x right, +y down, +z forward), pinhole. `K` calibrated if supplied, otherwise from `fov_deg` (horizontal FOV, centred principal point, `calibrated:false`).
- **Pose**: `X_cam = scale · R(rvec) · (X_model − center) + t` (mm). `center` = arch centroid pivot. `model_to_camera` is the equivalent 4×4. `R_FRONT` = camera in front of incisors.

## Person 1 → Person 3 (annotations JSON; also `data/samples/synthetic_annotations.json`)
```json
{ "markers":      [{"id": "M1", "image_xy": [x, y], "world_xyz": [x, y, z]}],
  "landmarks":    [{"fdi": 11, "image_xy": [x, y]}],
  "mask_polygons": [[[x, y], ...]] }
```
`world_xyz` is optional and currently stored only. `landmarks` pair an FDI tooth with its image position (model point = that tooth's centroid). Points must lie inside the image. Missing items are fine; automatic alignment then needs ≥4 landmarks or a polygon, otherwise only manual alignment is possible.
Optional camera calibration: `camera: {"K": [[fx,0,cx],[0,fy,cy],[0,0,1]]}` at registration creation.

## Person 2 → Person 3
Folder of `tooth_<FDI>.obj` (+ optional `metadata.json` with `jaw`) under a model root, or an uploaded single OBJ/PLY/STL (`≤100 MB`, geometry validated: non-empty, finite, valid faces). Git LFS pointers are rejected with a clear message.

## Alignment JSON (`dental-alignment/1`, written by export, reloadable)
`schema, status, mode (initial|auto|manual), disclaimer, image{id,name,width,height}, model{id,name,jaw,units,teeth}, camera{K,width,height,calibrated,projection,fov_deg?}, pose{rvec,t,scale,center,model_to_camera}, tooth_offsets{fdi:{t:[mm],r:[deg xyz about tooth centroid]}}, annotations, metrics{landmark_rmse_px?, silhouette_iou?}, method (pnp|mask_bbox|manual), notes[], config, conventions, exported_at`.
Metrics are present only when computable from the provided annotations; none are invented. Aligned meshes (`aligned_model.obj/.ply`) are in camera coordinates (mm), tooth offsets applied.

## API (FastAPI; interactive docs at `/docs`)
Success: `{"ok": true, ...}`; error: `{"ok": false, "error": "..."}` with 400 (bad input), 404 (unknown id).

| Method & path | Purpose |
|---|---|
| `GET /api/health` | liveness |
| `GET /api/models` | available models (`case:<root>/<name>` or `upload:<id>`) |
| `POST /api/models` (multipart `file`) | upload OBJ/PLY/STL |
| `GET /api/models/{kind}/{ref}/mesh?target_faces=` | decimated display mesh, per-face FDI labels, tooth centroids, `tooth_editing` flag |
| `POST /api/images` (multipart `file`) | upload PNG/JPEG (≤20 MB), returns size |
| `GET /api/images/{id}/file` | the photo |
| `GET /api/samples/{name}` | synthetic demo files |
| `POST /api/registrations` | `{image_id, model_id, camera?, annotations?}` → registration with the initial (frontal) pose |
| `GET /api/registrations/{id}` | state |
| `PUT /api/registrations/{id}/annotations` | replace annotations (validated against image bounds) |
| `POST /api/registrations/{id}/auto` | run alignment; body `{config:{...RegConfig}}`; unknown options rejected |
| `POST /api/registrations/{id}/transform` | manual pose (`rvec` or 3×3 `R`, `t`, `scale`, `center`) and/or `tooth_offsets` |
| `POST /api/registrations/{id}/export` | write `alignment.json`, `aligned_model.obj/.ply` |
| `GET /api/registrations/{id}/export/{file}` | download an export |
| `POST /api/registrations/load` | `{alignment, image_id}` → restore a saved alignment |

`RegConfig` keys: `w_landmark, w_model_to_mask, w_mask_to_model, w_scale_prior, loss, f_scale, max_rot_deg, scale_bounds, n_model_pts, n_mask_pts, max_nfev, min_iou_gain, fix_scale, polish_iou`.
