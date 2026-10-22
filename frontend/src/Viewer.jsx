import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

const PALETTE = [0xf1ece4, 0xe8e2d6, 0xf4efe8, 0xe3ddd0]

// Interactive viewer for one or two arches (upper/lower shown separately, never assumed to occlude).
export default function Viewer({ meshes, wireframe, selected }) {
  const host = useRef(null)
  const api = useRef({})

  useEffect(() => {
    const el = host.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1d2024)
    const camera = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, 1, 2000)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(el.clientWidth, el.clientHeight)
    el.appendChild(renderer.domElement)
    scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.1))
    const dl = new THREE.DirectionalLight(0xffffff, 1.2); dl.position.set(50, -80, -100); scene.add(dl)
    const controls = new OrbitControls(camera, renderer.domElement)
    const group = new THREE.Group(); scene.add(group)
    let raf
    const loop = () => { controls.update(); renderer.render(scene, camera); raf = requestAnimationFrame(loop) }
    loop()
    api.current = { scene, camera, controls, group, renderer }
    return () => { cancelAnimationFrame(raf); renderer.dispose(); el.removeChild(renderer.domElement) }
  }, [])

  useEffect(() => {
    const { group, camera, controls } = api.current
    group.clear()
    let offset = 0
    meshes.forEach((m, mi) => {
      const g = new THREE.BufferGeometry()
      const V = Float32Array.from(m.vertices)
      // un-indexed so each tooth can be coloured per face
      const n = m.faces.length
      const pos = new Float32Array(n * 3), col = new Float32Array(n * 3)
      for (let f = 0; f < m.face_tooth.length; f++) {
        const tooth = m.face_tooth[f]
        const c = new THREE.Color(String(tooth) === String(selected) ? 0xff7a1a : PALETTE[tooth % PALETTE.length])
        for (let k = 0; k < 3; k++) {
          const vi = m.faces[f * 3 + k]
          pos.set([V[vi * 3], V[vi * 3 + 1], V[vi * 3 + 2]], (f * 3 + k) * 3)
          col.set([c.r, c.g, c.b], (f * 3 + k) * 3)
        }
      }
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
      g.setAttribute('color', new THREE.BufferAttribute(col, 3))
      g.computeVertexNormals()
      const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, wireframe, side: THREE.DoubleSide, roughness: 0.5 }))
      // two arches are placed side by side only for display; each keeps its own coordinates
      mesh.position.set(-m.center[0] + mi * 90, -m.center[1], -m.center[2])
      group.add(mesh)
      offset += 1
    })
    api.current.reset = () => { camera.position.set(0, -140, -40); camera.up.set(0, 0, -1); controls.target.set(offset > 1 ? 45 : 0, 0, 0); controls.update() }
    api.current.reset()
  }, [meshes, wireframe, selected])

  return (
    <div className="viewer">
      <div ref={host} style={{ width: '100%', height: '100%' }} />
      <button className="reset" onClick={() => api.current.reset?.()}>Reset view</button>
    </div>
  )
}
