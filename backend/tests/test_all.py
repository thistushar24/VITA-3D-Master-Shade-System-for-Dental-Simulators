import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation as Rot

from dental_reg import core
from dental_reg.core import InputError, Pose
from dental_reg.synth import render

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
UPPER = SAMPLES / "proc_upper_01"
W, H = 1280, 853


@pytest.fixture(scope="module")
def arch():
    return core.load_arch_dir(UPPER)


@pytest.fixture(scope="module")
def K():
    return core.intrinsics(W, H)


def gt_pose(arch, K):
    base = core.initial_pose(arch, W, H, K)
    return Pose((Rot.from_euler("xyz", [5, 8, 3], degrees=True) * Rot.from_rotvec(base.rvec)).as_rotvec(),
                base.t + [6, -4, 10], 1.0, base.center)


# 1 photo loading
def test_photo_loading(tmp_path):
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.zeros((10, 20, 3), np.uint8))
    assert core.load_image(p).shape == (10, 20, 3)
    with pytest.raises(InputError, match="not found"):
        core.load_image(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(InputError, match="decoded"):
        core.load_image(bad)
    (tmp_path / "x.bmp").write_bytes(b"BM")
    with pytest.raises(InputError, match="Unsupported"):
        core.load_image(tmp_path / "x.bmp")


# 2 markers
def test_marker_validation():
    assert core.validate_points([[1, 2], [10, 5]], 20, 10).shape == (2, 2)
    with pytest.raises(InputError, match="outside image"):
        core.validate_points([[25, 2]], 20, 10)
    with pytest.raises(InputError, match="non-finite"):
        core.validate_points([[float("nan"), 2]], 20, 10)


# 3 + 9 mesh loading, invalid files
def test_mesh_loading(arch, tmp_path):
    assert arch.jaw == "upper" and arch.has_teeth and sorted(arch.teeth)[0] == 11
    assert len(arch.faces) == len(arch.face_tooth)
    junk = tmp_path / "m.obj"
    junk.write_text("hello world")
    with pytest.raises(InputError):
        core.load_mesh(junk)
    with pytest.raises(InputError, match="Unsupported"):
        core.load_mesh(tmp_path / "m.fbx")
    lfs = tmp_path / "lfs.obj"
    lfs.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
    with pytest.raises(InputError, match="LFS"):
        core.load_mesh(lfs)
    nan = tmp_path / "nan.obj"
    nan.write_text("v 0 0 0\nv 1 0 0\nv nan 1 0\nf 1 2 3\n")
    with pytest.raises(InputError, match="NaN"):
        core.load_mesh(nan)
    empty = tmp_path / "e.ply"
    trimesh.Trimesh().export(empty)
    with pytest.raises(InputError):
        core.load_mesh(empty)


# 4 + 6 coordinate transforms / matrix correctness
def test_pose_matrix_matches_projection(arch, K):
    p = gt_pose(arch, K)
    M = p.matrix()
    X = arch.vertices[:50]
    Xc = X @ M[:3, :3].T + M[:3, 3]
    uv_m = np.c_[K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]]
    uv, _, valid = core.project(X, p, K)
    assert valid.all() and np.allclose(uv, uv_m)
    assert np.allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3) * p.scale**2)
    assert core.Pose.from_json(p.to_json()).matrix() == pytest.approx(M)


def test_pose_from_rotation_matrix(arch, K):
    p = gt_pose(arch, K)
    d = p.to_json(); d.pop("rvec"); d["R"] = p.R().tolist()
    assert np.allclose(Pose.from_json(d).matrix(), p.matrix())


def test_original_mesh_untouched(arch, K):
    before = arch.vertices.copy()
    core.apply_tooth_offsets(arch, {"11": {"t": [1, 2, 3]}})
    assert np.array_equal(before, arch.vertices)


# 5 projection with known synthetic points
def test_projection_known_points():
    K = core.intrinsics(640, 480, K=[[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    p = Pose(np.zeros(3), np.array([0, 0, 100.0]), 1.0, np.zeros(3))
    uv, z, valid = core.project(np.array([[0, 0, 0], [10, 0, 0], [0, -20, 0], [0, 0, -150]]), p, K)
    assert np.allclose(uv[:3], [[320, 240], [370, 240], [320, 140]])
    assert list(valid) == [True, True, True, False]  # last point is behind the camera


# 7 + 8 synthetic registration, missing inputs
def test_registration_recovers_pose_from_mask(arch, K):
    gt = gt_pose(arch, K)
    img, mask = render(arch, gt, K, W, H)
    pose, rep = core.register(arch, img.shape, K, mask=mask)
    rot_err = np.degrees((Rot.from_rotvec(pose.rvec) * Rot.from_rotvec(gt.rvec).inv()).magnitude())
    assert rep["status"] == "ok" and rep["method"] == "mask_bbox"
    assert rot_err < 2.0, rot_err  # documented tolerance: 2 deg, 2 mm, IoU > 0.8
    assert np.linalg.norm(pose.t - gt.t) < 2.0
    assert rep["metrics"]["silhouette_iou"] > 0.8
    assert rep["metrics"]["silhouette_iou"] > rep["initial_metrics"]["silhouette_iou"]


def test_registration_pnp_exact(arch, K):
    gt = gt_pose(arch, K)
    lm = [(f, core.project(arch.teeth[f][None], gt, K)[0][0]) for f in (11, 21, 13, 23, 16, 26)]
    pose, rep = core.register(arch, (H, W), K, landmarks=lm)
    assert rep["method"] == "pnp" and rep["metrics"]["landmark_rmse_px"] < 1e-3
    assert np.allclose(pose.matrix(), gt.matrix(), atol=1e-4)


def test_registration_missing_inputs(arch, K):
    with pytest.raises(InputError, match="mask"):
        core.register(arch, (H, W), K)
    with pytest.raises(InputError, match="mask"):
        core.register(arch, (H, W), K, landmarks=[(11, (100, 100)), (21, (200, 100))])
    with pytest.raises(InputError, match="not in the model"):
        core.register(arch, (H, W), K, landmarks=[(99, (1, 1))], mask=np.full((H, W), 255, np.uint8))
    with pytest.raises(InputError, match="empty"):
        core.register(arch, (H, W), K, mask=np.zeros((H, W), np.uint8))


def test_bad_option_and_bad_pose():
    with pytest.raises(InputError):
        core.RegConfig.from_json({"nope": 1})
    with pytest.raises(InputError):
        Pose.from_json({"rvec": [0, 0, 0], "t": [0, 0, 1], "scale": 50, "center": [0, 0, 0]})


# tooth editing
def test_tooth_offsets(arch, tmp_path):
    V = core.apply_tooth_offsets(arch, {"11": {"t": [1, 0, 0]}})
    idx = np.unique(arch.faces[arch.face_tooth == 11])
    assert np.allclose(V[idx] - arch.vertices[idx], [1, 0, 0])
    other = np.setdiff1d(np.arange(len(V)), idx)
    assert np.array_equal(V[other], arch.vertices[other])
    single = core.load_arch_file(next(UPPER.glob("tooth_11.obj")))
    with pytest.raises(InputError, match="separate tooth"):
        core.apply_tooth_offsets(single, {"11": {}})


# 10 + 11 API, export and reload
@pytest.fixture()
def client(tmp_path, monkeypatch):
    from dental_reg import app as appmod
    monkeypatch.setattr(appmod, "STORAGE", tmp_path / "store")
    appmod.REGS.clear(); appmod.IMAGES.clear()
    (tmp_path / "store").mkdir()
    with TestClient(appmod.app) as c:
        yield c


def test_api_validation(client, tmp_path):
    assert client.get("/api/health").json()["ok"]
    r = client.post("/api/images", files={"file": ("x.bmp", b"abc")})
    assert r.status_code == 400 and "unsupported" in r.json()["error"]
    r = client.post("/api/images", files={"file": ("x.png", b"not a png")})
    assert r.status_code == 400 and "decoded" in r.json()["error"]
    r = client.post("/api/models", files={"file": ("x.obj", b"junk")})
    assert r.status_code == 400
    r = client.post("/api/models", files={"file": ("x.stl", b"")})
    assert r.status_code == 400
    assert client.post("/api/registrations", json={"image_id": "nope", "model_id": "x"}).status_code == 400
    assert client.get("/api/registrations/nope").status_code == 404
    assert any(m["id"].endswith("proc_upper_01") for m in client.get("/api/models").json()["models"])
    assert client.get("/api/models/case/samples/..%2F..%2Fetc/mesh").status_code == 404


def test_api_full_flow_export_reload(client):
    img = client.post("/api/images", files={"file": ("p.png", (SAMPLES / "synthetic_photo.png").read_bytes())}).json()
    assert (img["width"], img["height"]) == (W, H)
    mesh = client.get("/api/models/case/samples/proc_upper_01/mesh", params={"target_faces": 3000}).json()
    assert mesh["tooth_editing"] and len(mesh["faces"]) % 3 == 0
    ann = json.loads((SAMPLES / "synthetic_annotations.json").read_text())
    ann.pop("_note")
    reg = client.post("/api/registrations", json={"image_id": img["id"], "model_id": "case:samples/proc_upper_01",
                                                    "annotations": ann}).json()
    assert reg["camera"]["calibrated"] is False  # approximate camera is labelled as such
    out = client.post(f"/api/registrations/{reg['id']}/auto", json={"config": {}}).json()
    assert out["report"]["method"] == "pnp" and out["report"]["metrics"]["landmark_rmse_px"] < 0.2  # annotations are rounded to 0.1 px
    # manual tweak + tooth edit
    pose = out["pose"]; pose["t"][0] += 1.0
    man = client.post(f"/api/registrations/{reg['id']}/transform",
                      json={"pose": pose, "tooth_offsets": {"11": {"t": [0.5, 0, 0], "r": [0, 0, 5]}}}).json()
    assert man["mode"] == "manual" and man["report"]["metrics"]["landmark_rmse_px"] > 1
    ex = client.post(f"/api/registrations/{reg['id']}/export").json()
    doc = ex["alignment"]
    assert doc["schema"] == "dental-alignment/1" and "not a reconstruction" in doc["disclaimer"]
    obj = client.get(f"/api/registrations/{reg['id']}/export/aligned_model.obj")
    assert obj.status_code == 200
    m = trimesh.load(__import__("io").BytesIO(obj.content), file_type="obj", force="mesh", process=False)
    assert len(m.vertices) == len(core.load_arch_dir(UPPER).vertices)
    # reload
    back = client.post("/api/registrations/load", json={"alignment": doc, "image_id": img["id"]}).json()
    assert np.allclose(back["pose"]["model_to_camera"], doc["pose"]["model_to_camera"])
    assert back["tooth_offsets"] == doc["tooth_offsets"]
    # originals untouched
    assert not list(UPPER.glob("*.ply"))


def test_api_auto_without_inputs(client):
    img = client.post("/api/images", files={"file": ("p.png", (SAMPLES / "synthetic_photo.png").read_bytes())}).json()
    reg = client.post("/api/registrations", json={"image_id": img["id"], "model_id": "case:samples/proc_upper_01"}).json()
    r = client.post(f"/api/registrations/{reg['id']}/auto", json={})
    assert r.status_code == 400 and "mask" in r.json()["error"]
    # landmark outside image rejected
    r = client.put(f"/api/registrations/{reg['id']}/annotations", json={"landmarks": [{"fdi": 11, "image_xy": [5000, 5]}]})
    assert r.status_code == 400


# ---- v1: library, segmentation, matching
LIB = Path(__file__).resolve().parents[2] / "data" / "demo_library"


def test_library_catalogue_and_npz():
    from dental_reg import library
    cat = library.load_catalogue(LIB)
    assert len(cat) >= 10 and all(e["license"] and e["source"] for e in cat)
    e = next(x for x in cat if x["classification"] == "complete_arch")
    a = library.load_arch_npz(LIB / e["file"])
    assert a.has_teeth and 13 in a.teeth and e["features"]["canine_distance_mm"] > 15
    with pytest.raises(InputError, match="missing"):
        library.load_arch_npz(LIB / "nope.npz")
    with pytest.raises(InputError, match="catalogue"):
        library.load_catalogue(LIB / "nowhere")


def test_segmentation_on_synthetic_photo(arch, K):
    from dental_reg import vision
    gt = gt_pose(arch, K)
    img, true = render(arch, gt, K, W, H, tooth=(170, 215, 240), bg=(40, 30, 150))  # warm teeth on red 'gums'
    seg = vision.segment_teeth(img)
    assert core.iou(seg["mask_all"], true) > 0.6
    assert seg["confidence"] == "preliminary" and len(seg["canines"]) == 2
    flat = np.full((200, 300, 3), (60, 60, 200), np.uint8)
    with pytest.raises(InputError, match="manually"):
        vision.segment_teeth(flat)


def test_match_is_reproducible_and_finds_source_model(K):
    """Sanity check only: the model a synthetic photo was rendered from should rank first."""
    from dental_reg import library, match
    cat = library.load_catalogue(LIB)
    up = [e for e in cat if e["jaw"] == "upper" and e["classification"] == "complete_arch"][:6]
    src = library.load_arch_npz(LIB / up[2]["file"])
    base = core.initial_pose(src, W, H, K)
    gt = Pose((Rot.from_euler("xyz", [3, 5, 2], degrees=True) * Rot.from_rotvec(base.rvec)).as_rotvec(), base.t + [4, -3, 6], 1.0, base.center)
    _, mask = render(src, gt, K, W, H)
    can = core.project(np.array([src.teeth[13], src.teeth[23]]), gt, K)[0].tolist()
    args = (mask, can, up, LIB, (H, W, 3))
    r1, r2 = match.rank_models(*args), match.rank_models(*args)
    assert r1["selected"] == up[2]["id"]
    assert r1["selected"] == r2["selected"] and r1["ranked"][0]["final"] == r2["ranked"][0]["final"]
    assert "Scores are hand-weighted" in r1["note"] and r1["ranked"][0]["fit_iou"] > 0.7  # ranking uses the fast single-resolution polish
    with pytest.raises(InputError, match="canine"):
        match.rank_models(mask, None, up, LIB, (H, W, 3))
    with pytest.raises(InputError, match="no lower"):
        match.rank_models(mask, can, up, LIB, (H, W, 3), jaw="lower")


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_v1_full_journey(client, K):
    from dental_reg import library
    cat = library.load_catalogue(LIB)
    up = [e for e in cat if e["jaw"] == "upper" and e["classification"] == "complete_arch"][:6]
    src = library.load_arch_npz(LIB / up[1]["file"])
    base = core.initial_pose(src, W, H, K)
    gt = Pose((Rot.from_euler("xyz", [3, 5, 2], degrees=True) * Rot.from_rotvec(base.rvec)).as_rotvec(), base.t + [4, -3, 6], 1.0, base.center)
    photo_a, _ = render(src, gt, K, W, H, tooth=(170, 215, 240), bg=(40, 30, 150))
    photo_b = np.full((600, 800, 3), (120, 140, 200), np.uint8)
    ia = client.post("/api/images", files={"file": ("a.png", _png(photo_a))}).json()
    ib = client.post("/api/images", files={"file": ("b.png", _png(photo_b))}).json()
    cat_r = client.get("/api/v1/catalogue").json()
    assert cat_r["count"] >= 10
    s = client.post("/api/v1/sessions", json={"photo_a_id": ia["id"], "photo_b_id": ib["id"]}).json()
    assert s["analysis"]["mask_polygons"] and s["analysis"]["confidence"] == "preliminary"
    # use true canine positions (what a user would drag the points to)
    can = core.project(np.array([src.teeth[13], src.teeth[23]]), gt, K)[0].tolist()
    assert client.put(f"/api/v1/sessions/{s['id']}/annotations", json={"canines": [[-5, 2], [3, 3]]}).status_code == 400
    client.put(f"/api/v1/sessions/{s['id']}/annotations", json={"canines": can})
    # editing before confirmation is refused
    assert client.post(f"/api/v1/sessions/{s['id']}/versions", json={}).status_code == 400
    m = client.post(f"/api/v1/sessions/{s['id']}/match", json={}).json()
    assert m["match"]["selected"] in {e["id"] for e in up + cat} and m["registration"]["model"]["name"]
    assert m["model_id"] == "lib:" + m["match"]["selected"] and m["match"]["explanation"]
    # replace with a different catalogue model, then back
    other = next(e["id"] for e in up if e["id"] != m["match"]["selected"])
    assert client.post(f"/api/v1/sessions/{s['id']}/replace_model", json={"model_id": other}).json()["model_id"] == "lib:" + other
    assert client.post(f"/api/v1/sessions/{s['id']}/replace_model", json={"model_id": "../../x"}).status_code == 400
    c = client.post(f"/api/v1/sessions/{s['id']}/confirm_alignment", json={}).json()
    assert c["stage"] == "editing" and c["alignment_a"]["pose"]["model_to_camera"] and len(c["versions"]) == 1
    v = client.post(f"/api/v1/sessions/{s['id']}/versions", json={"tooth_offsets": {"11": {"t": [0.4, 0, 0]}}, "note": "nudge 11"}).json()
    assert [x["n"] for x in v["versions"]] == [1, 2]
    assert client.post(f"/api/v1/sessions/{s['id']}/versions", json={"tooth_offsets": {"99": {}}}).status_code == 400
    v = client.post(f"/api/v1/sessions/{s['id']}/versions/1/restore").json()
    assert v["versions"][-1]["note"] == "restored from v1"
    pb = client.put(f"/api/v1/sessions/{s['id']}/photo_b", json={"crop": [100, 100, 400, 200], "pose": c["alignment_a"]["pose"]}).json()
    assert pb["stage"] == "simulating" and pb["photo_b_placement"]["crop"] == [100, 100, 400, 200]
    assert client.put(f"/api/v1/sessions/{s['id']}/photo_b", json={"crop": [0, 0, 5000, 200]}).status_code == 400
    # Photo A and B poses stay separate; working copy exists, original library file unchanged
    from dental_reg import app as appmod
    assert (appmod.STORAGE / f"session_{s['id']}" / "working_model.npz").is_file()
    # the library-model mesh endpoint serves the selected model
    assert client.get(f"/api/models/lib/{other}/mesh").json()["tooth_editing"] is True
    assert client.get("/api/models/lib/not_a_model/mesh").status_code == 404


def test_state_survives_restart(client, K):
    """Sessions, registrations and uploads are reloaded from disk after the in-memory state is lost."""
    from dental_reg import app as appmod, library
    cat = library.load_catalogue(LIB)
    src = library.load_arch_npz(LIB / cat[0]["file"])
    base = core.initial_pose(src, W, H, K)
    img, _ = render(src, base, K, W, H, tooth=(170, 215, 240), bg=(40, 30, 150))
    ia = client.post("/api/images", files={"file": ("a.png", _png(img))}).json()
    s = client.post("/api/v1/sessions", json={"photo_a_id": ia["id"]}).json()
    can = core.project(np.array([src.teeth[13], src.teeth[23]]), base, K)[0].tolist()
    client.put(f"/api/v1/sessions/{s['id']}/annotations", json={"canines": can})
    m = client.post(f"/api/v1/sessions/{s['id']}/match", json={}).json()
    client.post(f"/api/v1/sessions/{s['id']}/confirm_alignment", json={})
    appmod.REGS.clear(); appmod.IMAGES.clear(); v1mod_sessions().clear()  # simulate a server restart
    s2 = client.get(f"/api/v1/sessions/{s['id']}").json()
    assert s2["stage"] == "editing" and s2["model_id"] == m["model_id"]
    r2 = client.get(f"/api/registrations/{m['registration']['id']}").json()
    assert np.allclose(r2["pose"]["model_to_camera"], m["registration"]["pose"]["model_to_camera"])
    assert client.get(f"/api/images/{ia['id']}/file").status_code == 200
    v = client.post(f"/api/v1/sessions/{s['id']}/versions", json={"tooth_offsets": {}, "note": "after restart"}).json()
    assert [x["n"] for x in v["versions"]] == [1, 2]


def v1mod_sessions():
    from dental_reg import v1
    return v1.SESSIONS


def test_silhouette_is_union_not_even_odd():
    """Regression: cv2.fillPoly cancels overlapping triangles (even-odd). Front+back faces of a closed tooth overlap."""
    K2 = core.intrinsics(200, 200, K=[[200, 0, 100], [0, 200, 100], [0, 0, 1]])
    V = np.array([[-10, -10, 0], [10, -10, 0], [0, 10, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 1]])  # same triangle twice (front and back face)
    p = Pose(np.zeros(3), np.array([0, 0, 100.0]), 1.0, np.zeros(3))
    one = core.silhouette(V, F[:1], p, K2, 200, 200)
    two = core.silhouette(V, F, p, K2, 200, 200)
    assert one.sum() > 0 and np.array_equal(one, two)
    labelled = core.silhouette(V, F, p, K2, 200, 200, face_tooth=np.array([11, 12]))
    assert labelled.sum() > 0


def test_canine_estimate_geometry():
    from dental_reg import vision
    m = np.zeros((300, 1000), np.uint8)
    m[100:200, 100:900] = 255  # symmetric 800 px wide tooth band
    (l, r), _ = vision.estimate_canines(m)
    assert abs(l[0] - (500 - 0.33 * 800)) < 2 and abs(r[0] - (500 + 0.33 * 800)) < 2
    assert 140 < l[1] < 160 and l[0] < r[0]  # patient's right (image-left) first
