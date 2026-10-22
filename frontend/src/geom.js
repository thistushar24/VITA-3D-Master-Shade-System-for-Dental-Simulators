// Pose/projection maths. Must mirror backend/dental_reg/core.py:
//   X_cam = scale * R @ (X_model - center) + t ;  u = fx*x/z + cx ,  v = fy*y/z + cy
// Matrices are row-major 3x3 arrays [[..],[..],[..]].
export const mul = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]))
export const apply = (R, v) => [0, 1, 2].map((i) => R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2])

export function rodrigues([x, y, z]) {
  const th = Math.hypot(x, y, z)
  if (th < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  const [kx, ky, kz] = [x / th, y / th, z / th]
  const c = Math.cos(th), s = Math.sin(th), C = 1 - c
  return [
    [c + kx * kx * C, kx * ky * C - kz * s, kx * kz * C + ky * s],
    [ky * kx * C + kz * s, c + ky * ky * C, ky * kz * C - kx * s],
    [kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C],
  ]
}

// Euler XYZ in degrees about fixed axes (x first): R = Rz Ry Rx  (scipy 'xyz' extrinsic)
export function eulerXYZ([ax, ay, az]) {
  const d = Math.PI / 180
  const [cx, sx, cy, sy, cz, sz] = [Math.cos(ax * d), Math.sin(ax * d), Math.cos(ay * d), Math.sin(ay * d), Math.cos(az * d), Math.sin(az * d)]
  const Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
  const Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
  return mul(Rz, mul(Ry, Rx))
}

// backend pose JSON -> {R, t, scale, center}
export const poseFromJson = (p) => ({ R: rodrigues(p.rvec), t: p.t, scale: p.scale, center: p.center })

// manual deltas (camera-frame rotation in degrees, translation in mm, scale factor) on top of a base pose
export function composePose(base, d) {
  return {
    R: mul(eulerXYZ(d.rot), base.R),
    t: [base.t[0] + d.t[0], base.t[1] + d.t[1], base.t[2] + d.t[2]],
    scale: base.scale * d.scale,
    center: base.center,
  }
}

export const poseToJson = (p) => ({ R: p.R, t: p.t, scale: p.scale, center: p.center })

export function applyToothOffsets(V, faces, faceTooth, teeth, offsets) {
  const out = Float32Array.from(V)
  for (const [fdi, o] of Object.entries(offsets || {})) {
    const c = teeth[fdi]
    if (!c) continue
    const R = eulerXYZ(o.r || [0, 0, 0])
    const t = o.t || [0, 0, 0]
    const seen = new Set()
    for (let f = 0; f < faceTooth.length; f++) {
      if (String(faceTooth[f]) !== String(fdi)) continue
      for (let k = 0; k < 3; k++) {
        const i = faces[f * 3 + k]
        if (seen.has(i)) continue
        seen.add(i)
        const p = apply(R, [out[i * 3] - c[0], out[i * 3 + 1] - c[1], out[i * 3 + 2] - c[2]])
        for (let a = 0; a < 3; a++) out[i * 3 + a] = p[a] + c[a] + t[a]
      }
    }
  }
  return out
}

// returns {uv: Float32Array(2n), z: Float32Array(n)}; z <= near means invalid (behind camera)
export function projectVerts(V, pose, K, near = 1) {
  const n = V.length / 3
  const uv = new Float32Array(n * 2), z = new Float32Array(n)
  const { R, t, scale, center } = pose
  for (let i = 0; i < n; i++) {
    const x = V[i * 3] - center[0], y = V[i * 3 + 1] - center[1], zz = V[i * 3 + 2] - center[2]
    const X = scale * (R[0][0] * x + R[0][1] * y + R[0][2] * zz) + t[0]
    const Y = scale * (R[1][0] * x + R[1][1] * y + R[1][2] * zz) + t[1]
    const Z = scale * (R[2][0] * x + R[2][1] * y + R[2][2] * zz) + t[2]
    z[i] = Z
    if (Z > near) { uv[i * 2] = (K[0][0] * X) / Z + K[0][2]; uv[i * 2 + 1] = (K[1][1] * Y) / Z + K[1][2] }
  }
  return { uv, z }
}
