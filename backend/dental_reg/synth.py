"""Synthetic 'photograph' renderer for tests/demo. Output is NOT a real dental photo."""
import cv2
import numpy as np

from .core import Arch, Pose, project, silhouette


def render(arch: Arch, pose: Pose, K, width, height, V=None, tooth=(235, 240, 245), bg=(40, 30, 90)):
    """Painter's-algorithm shaded render of the arch. Returns (BGR image, binary silhouette mask)."""
    V = arch.vertices if V is None else V
    uv, z, valid = project(V, pose, K)
    F = arch.faces
    ok = valid[F].all(1)
    F, ft = F[ok], arch.face_tooth[ok]
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    nc = n @ pose.R().T
    flip = nc[:, 2] > 0  # make normals face the camera for shading regardless of winding
    nc[flip] *= -1
    shade = np.clip(-nc[:, 2] * 0.8 + 0.25, 0.1, 1.0)
    order = np.argsort(-z[F].mean(1))  # far to near
    img = np.full((height, width, 3), bg, np.uint8)  # dark 'mouth'
    base = np.array(tooth, float)
    tint = 1 - 0.04 * (ft % 5)
    for i in order:
        col = tuple(int(c) for c in base * shade[i] * tint[i])
        cv2.fillConvexPoly(img, np.round(uv[F[i]]).astype(np.int32), col, cv2.LINE_AA)
    mask = silhouette(V, arch.faces, pose, K, width, height, arch.face_tooth)
    return img, mask
