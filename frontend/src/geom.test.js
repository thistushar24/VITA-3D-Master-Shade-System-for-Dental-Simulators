import { describe, it, expect } from 'vitest'
import { rodrigues, eulerXYZ, projectVerts, composePose, applyToothOffsets } from './geom.js'

const K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
const I = { R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 0, 100], scale: 1, center: [0, 0, 0] }

describe('geom', () => {
  it('projects known points and flags behind-camera ones (same numbers as the backend test)', () => {
    const { uv, z } = projectVerts(Float32Array.of(0, 0, 0, 10, 0, 0, 0, -20, 0, 0, 0, -150), I, K)
    expect([...uv.slice(0, 6)]).toEqual([320, 240, 370, 240, 320, 140])
    expect(z[3]).toBeLessThan(1)
  })
  it('rodrigues handles the 180 degree frontal rotation', () => {
    const R = rodrigues([Math.PI, 0, 0])
    expect(R[1][1]).toBeCloseTo(-1); expect(R[2][2]).toBeCloseTo(-1)
  })
  it('euler z 90deg maps x->y', () => {
    const R = eulerXYZ([0, 0, 90])
    expect(R[1][0]).toBeCloseTo(1); expect(R[0][1]).toBeCloseTo(-1)
  })
  it('composePose applies translation, scale, and rotation about the pivot', () => {
    const p = composePose(I, { rot: [0, 0, 0], t: [1, 2, 3], scale: 2 })
    expect(p.t).toEqual([1, 2, 103]); expect(p.scale).toBe(2)
  })
  it('tooth offset moves only that tooth', () => {
    const V = Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 5, 5, 6, 5, 5, 5, 6, 5)
    const out = applyToothOffsets(V, [0, 1, 2, 3, 4, 5], [11, 12], { 11: [0, 0, 0] }, { 11: { t: [1, 0, 0] } })
    expect([...out.slice(0, 3)]).toEqual([1, 0, 0]); expect([...out.slice(9, 12)]).toEqual([5, 5, 5])
  })
})
