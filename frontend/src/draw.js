import { projectVerts, apply } from './geom.js'

// painter's-algorithm render of the mesh onto a 2D canvas (faces sorted far to near)
export function drawMesh(ctx, mesh, V, pose, K, { alpha = 0.6, selected = null, wire = false, color = [90, 170, 255] }) {
  const { uv, z } = projectVerts(V, pose, K)
  const F = mesh.faces, order = []
  for (let f = 0; f < F.length / 3; f++) {
    const a = F[f * 3], b = F[f * 3 + 1], c = F[f * 3 + 2]
    if (z[a] > 1 && z[b] > 1 && z[c] > 1) order.push([f, (z[a] + z[b] + z[c]) / 3])
  }
  order.sort((p, q) => q[1] - p[1])
  ctx.save()
  ctx.globalAlpha = alpha
  ctx.lineWidth = 0.6
  for (const [f] of order) {
    const [a, b, c] = [F[f * 3], F[f * 3 + 1], F[f * 3 + 2]]
    const e1 = [V[b * 3] - V[a * 3], V[b * 3 + 1] - V[a * 3 + 1], V[b * 3 + 2] - V[a * 3 + 2]]
    const e2 = [V[c * 3] - V[a * 3], V[c * 3 + 1] - V[a * 3 + 1], V[c * 3 + 2] - V[a * 3 + 2]]
    const n = apply(pose.R, [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]])
    const len = Math.hypot(...n) || 1
    const sh = Math.min(1, Math.max(0.15, Math.abs(n[2]) / len * 0.8 + 0.2))
    const sel = String(mesh.face_tooth[f]) === String(selected)
    ctx.fillStyle = sel ? `rgb(${255 * sh},${120 * sh},${20 * sh})` : `rgb(${color[0] * sh},${color[1] * sh},${color[2] * sh})`
    ctx.strokeStyle = 'rgba(0,40,90,.5)'
    ctx.beginPath()
    ctx.moveTo(uv[a * 2], uv[a * 2 + 1]); ctx.lineTo(uv[b * 2], uv[b * 2 + 1]); ctx.lineTo(uv[c * 2], uv[c * 2 + 1]); ctx.closePath()
    if (!wire) ctx.fill()
    if (wire || sel) ctx.stroke()
  }
  ctx.restore()
}

