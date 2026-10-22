import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Viewer from '@shared/Viewer.jsx'
import { drawMesh } from '@shared/draw.js'
import { applyToothOffsets, composePose, poseFromJson, poseToJson, projectVerts } from '@shared/geom.js'
import { v1 } from './api.js'

const STAGES = ['1 Upload', '2 Auto-match', '3 Align on Photo A', '4 Edit 3D model', '5 Simulate on Photo B']
const ZERO = () => ({ rot: [0, 0, 0], t: [0, 0, 0], scale: 1 })
const f2 = (x, n = 2) => (typeof x === 'number' ? x.toFixed(n) : '—')
const IVORY = [236, 228, 212]

function useImage(url) {
  const [img, setImg] = useState(null)
  useEffect(() => {
    if (!url) return setImg(null)
    const i = new Image()
    i.onload = () => setImg(i)
    i.src = url
  }, [url])
  return img
}

function Slider({ label, value, min, max, step, onChange, onStart, unit = '' }) {
  return (
    <>
      <label>{label}</label>
      <input type="range" min={min} max={max} step={step} value={value} onPointerDown={onStart} onKeyDown={onStart}
        onChange={(e) => onChange(+e.target.value)} />
      <span className="muted">{f2(value)}{unit}</span>
    </>
  )
}

const toImg = (e, canvas, w, h) => {
  const r = canvas.getBoundingClientRect()
  return [((e.clientX - r.left) * w) / r.width, ((e.clientY - r.top) * h) / r.height]
}
const pinhole = (w, h, fov) => {
  const f = w / 2 / Math.tan((fov * Math.PI) / 360)
  return [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]
}

export default function App() {
  const [stage, setStage] = useState(0)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState('')
  const [photoA, setPhotoA] = useState(null)
  const [photoB, setPhotoB] = useState(null)
  const [session, setSession] = useState(null)
  const [catalogue, setCatalogue] = useState([])
  const [reg, setReg] = useState(null)
  const demoCanines = useRef(null)
  const [mesh, setMesh] = useState(null)
  const imgA = useImage(photoA?.url)
  const imgB = useImage(photoB?.url)

  const guard = useCallback(async (label, fn) => {
    setErr(''); setBusy(label)
    try { return await fn() } catch (e) { setErr(e.message) } finally { setBusy('') }
  }, [])

  useEffect(() => { v1.catalogue().then((r) => setCatalogue(r.models)).catch((e) => setErr(`Model catalogue unavailable: ${e.message}`)) }, [])

  // ---------------- stage 1: upload
  const upload = (which) => (f) => f && guard('Uploading…', async () => {
    const r = await v1.uploadImage(f, which)
    const p = { ...r, url: URL.createObjectURL(r.blob || f) } // r.blob = the (possibly downscaled) image the coordinates refer to
    which === 'A' ? setPhotoA(p) : setPhotoB(p)
    if (which === 'A') demoCanines.current = null
    setSession(null); setReg(null); setMesh(null)
  })
  const demo = () => guard('Loading demo…', async () => {
    for (const [n, set, role] of [['demo_photo_a.png', setPhotoA, 'A'], ['demo_photo_b.png', setPhotoB, 'B']]) {
      const blob = await v1.sample(n)
      const r = await v1.uploadImage(new File([blob], n, { type: 'image/png' }), role)
      set({ ...r, url: URL.createObjectURL(r.blob || blob) })
    }
    demoCanines.current = JSON.parse(await (await v1.sample('demo_annotations.json')).text()).canines
    setSession(null); setReg(null); setMesh(null)
  })
  const analyse = () => guard('Detecting tooth region…', async () => {
    let s = await v1.createSession(photoA.id, photoB?.id)
    if (demoCanines.current) s = await v1.setAnnotations(s.id, { canines: demoCanines.current }) // demo only: canines pre-placed
    setSession(s); setStage(1)
  })

  // ---------------- stage 2: canines + match
  const annRef = useRef()
  const dragging = useRef(-1)
  const [drawMode, setDrawMode] = useState(false)
  const [draft, setDraft] = useState([])
  useEffect(() => {
    if (stage !== 1 || !imgA || !annRef.current || !session) return
    const c = annRef.current.getContext('2d')
    c.drawImage(imgA, 0, 0)
    const a = session.analysis
    c.lineWidth = 3; c.strokeStyle = '#22d3ee'
    a.mask_polygons.forEach((p) => { c.beginPath(); p.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath(); c.stroke() })
    c.strokeStyle = '#fde047'; c.beginPath(); draft.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.stroke()
    ;(a.canines || []).forEach((p, i) => {
      c.fillStyle = '#facc15'; c.beginPath(); c.arc(p[0], p[1], 12, 0, 7); c.fill()
      c.fillStyle = '#000'; c.font = 'bold 16px sans-serif'; c.fillText(i ? '23' : '13', p[0] - 9, p[1] + 5)
    })
  }, [stage, imgA, session, draft])
  const onDown = (e) => {
    const [x, y] = toImg(e, annRef.current, photoA.width, photoA.height)
    if (drawMode) return setDraft([...draft, [Math.round(x), Math.round(y)]])
    const cs = session.analysis.canines || []
    dragging.current = cs.findIndex((p) => Math.hypot(p[0] - x, p[1] - y) < 30 * (photoA.width / annRef.current.getBoundingClientRect().width / 2))
    if (dragging.current >= 0) annRef.current.setPointerCapture(e.pointerId)
  }
  const onMove = (e) => {
    if (dragging.current < 0) return
    const [x, y] = toImg(e, annRef.current, photoA.width, photoA.height)
    const cs = session.analysis.canines.map((p, i) => (i === dragging.current ? [Math.max(0, Math.min(photoA.width - 1, x)), Math.max(0, Math.min(photoA.height - 1, y))] : p))
    setSession({ ...session, analysis: { ...session.analysis, canines: cs } })
  }
  const onUp = () => {
    if (dragging.current >= 0) { dragging.current = -1; guard('', async () => setSession(await v1.setAnnotations(session.id, { canines: session.analysis.canines }))) }
  }
  const closeDraft = () => guard('', async () => {
    setSession(await v1.setAnnotations(session.id, { mask_polygons: [...(drawMode ? [] : session.analysis.mask_polygons), draft] })); setDraft([]); setDrawMode(false)
  })
  const [fov, setFov] = useState(40)
  const runMatch = () => guard('Ranking models and fitting…', async () => {
    const r = await v1.match(session.id, { fov_deg: fov })
    setSession(r); setReg(r.registration)
    setMesh(await v1.mesh(r.model_id))
  })
  const replace = (id) => id && guard('Fitting replacement…', async () => {
    const r = await v1.replace(session.id, id)
    setSession(r); setReg(r.registration); setMesh(await v1.mesh(r.model_id))
  })

  // ---------------- stage 3: align on A
  const [d, setD] = useState(ZERO())
  const [hist, setHist] = useState({ past: [], future: [] })
  const [alpha, setAlpha] = useState(0.55)
  const [zoom, setZoom] = useState(1)
  useEffect(() => { if (reg) { setD(ZERO()); setHist({ past: [], future: [] }) } }, [reg?.id, reg?.pose?.model_to_camera?.[0]?.[0]])
  const baseA = useMemo(() => (reg ? poseFromJson(reg.pose) : null), [reg])
  const poseA = useMemo(() => (baseA ? composePose(baseA, d) : null), [baseA, d])
  const KA = reg?.camera.K
  const snap = () => setHist((h) => ({ past: [...h.past.slice(-50), d], future: [] }))
  const undo = () => setHist((h) => { if (!h.past.length) return h; setD(h.past[h.past.length - 1]); return { past: h.past.slice(0, -1), future: [d, ...h.future] } })
  const redo = () => setHist((h) => { if (!h.future.length) return h; setD(h.future[0]); return { past: [...h.past, d], future: h.future.slice(1) } })
  const alignRef = useRef()
  const V0 = useMemo(() => (mesh ? Float32Array.from(mesh.vertices) : null), [mesh])
  useEffect(() => {
    if (stage !== 2 || !imgA || !mesh || !poseA || !alignRef.current) return
    const c = alignRef.current.getContext('2d')
    c.clearRect(0, 0, photoA.width, photoA.height)
    c.drawImage(imgA, 0, 0)
    drawMesh(c, mesh, V0, poseA, KA, { alpha })
    const cs = session.analysis.canines || []
    cs.forEach((p, i) => { c.strokeStyle = '#facc15'; c.lineWidth = 3; c.beginPath(); c.moveTo(p[0] - 14, p[1]); c.lineTo(p[0] + 14, p[1]); c.moveTo(p[0], p[1] - 14); c.lineTo(p[0], p[1] + 14); c.stroke() })
    ;[13, 23].forEach((f) => {
      if (!mesh.teeth[f]) return
      const { uv } = projectVerts(Float32Array.from(mesh.teeth[f]), poseA, KA)
      c.strokeStyle = '#fb923c'; c.lineWidth = 3; c.beginPath(); c.arc(uv[0], uv[1], 11, 0, 7); c.stroke()
    })
  }, [stage, imgA, mesh, V0, poseA, KA, alpha, session])
  const dirty = d.scale !== 1 || d.rot.some(Boolean) || d.t.some(Boolean)
  const confirm = () => guard('Saving alignment…', async () => { setSession(await v1.confirm(session.id, dirty ? poseToJson(poseA) : null)); setStage(3) })

  // ---------------- stage 4: edit
  const [offsets, setOffsets] = useState({})
  const [sel, setSel] = useState('')
  const [note, setNote] = useState('')
  const Vedit = useMemo(() => (mesh ? applyToothOffsets(V0, mesh.faces, mesh.face_tooth, mesh.teeth, offsets) : null), [mesh, V0, offsets])
  const viewMeshes = useMemo(() => (mesh ? [{ ...mesh, vertices: Array.from(Vedit) }] : []), [mesh, Vedit])
  const saveVersion = () => guard('Saving version…', async () => { setSession(await v1.saveVersion(session.id, offsets, note)); setNote('') })
  const restore = (n) => guard('', async () => {
    const v = session.versions.find((x) => x.n === n)
    setSession(await v1.restore(session.id, n)); setOffsets(v.tooth_offsets || {})
  })

  // ---------------- stage 5: Photo B
  const [crop, setCrop] = useState(null) // [x,y,w,h] in Photo B px
  const [cropping, setCropping] = useState(true)
  const [fovB, setFovB] = useState(45)
  const [dB, setDB] = useState(ZERO())
  const [alphaB, setAlphaB] = useState(0.9)
  const [layer, setLayer] = useState('over') // over: polygons hide model ; under: model only visible inside polygons
  const [polys, setPolys] = useState([])
  const [draftB, setDraftB] = useState([])
  const [drawB, setDrawB] = useState(false)
  const [compare, setCompare] = useState(false)
  const [split, setSplit] = useState(50)
  const bRef = useRef(), cropRef = useRef(), dragCrop = useRef(null)
  const KB = useMemo(() => (photoB ? pinhole(photoB.width, photoB.height, fovB) : null), [photoB, fovB])
  const baseB = useMemo(() => {
    if (!mesh || !session?.alignment_a || !crop || !KB) return null
    const pa = poseFromJson(session.alignment_a.pose)
    let xmin = 1e9, xmax = -1e9
    for (let i = 0; i < mesh.vertices.length; i += 3) { xmin = Math.min(xmin, mesh.vertices[i]); xmax = Math.max(xmax, mesh.vertices[i]) }
    const tz = (KB[0][0] * (xmax - xmin)) / (0.7 * crop[2])
    const cxc = crop[0] + crop[2] / 2, cyc = crop[1] + crop[3] / 2
    return { R: pa.R, t: [((cxc - KB[0][2]) * tz) / KB[0][0], ((cyc - KB[1][2]) * tz) / KB[1][1], tz], scale: 1, center: pa.center }
  }, [mesh, session?.alignment_a, crop, KB])
  const poseB = useMemo(() => (baseB ? composePose(baseB, dB) : null), [baseB, dB])
  useEffect(() => { setDB(ZERO()) }, [baseB])

  const renderB = useCallback((cv, { guides, cmp }) => {
    const c = cv.getContext('2d')
    c.save(); c.translate(-crop[0], -crop[1])
    c.drawImage(imgB, 0, 0)
    const model = () => drawMesh(c, mesh, Vedit, poseB, KB, { alpha: alphaB, color: IVORY })
    const clipPolys = () => { c.beginPath(); polys.forEach((p) => { p.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath() }) }
    if (layer === 'over') {
      model()
      polys.forEach((p) => { c.save(); c.beginPath(); p.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath(); c.clip(); c.drawImage(imgB, 0, 0); c.restore() })
    } else if (polys.length) { c.save(); clipPolys(); c.clip(); model(); c.restore() }
    if (cmp) { c.save(); c.beginPath(); c.rect(crop[0] + (crop[2] * split) / 100, crop[1], crop[2], crop[3]); c.clip(); c.drawImage(imgB, 0, 0); c.restore() }
    if (guides) {
      c.strokeStyle = '#fde047'; c.lineWidth = 2
      c.beginPath(); draftB.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.stroke()
      c.strokeStyle = '#22d3ee'; c.setLineDash([6, 4]); polys.forEach((p) => { c.beginPath(); p.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath(); c.stroke() })
      if (cmp) { c.setLineDash([]); c.strokeStyle = '#fff'; c.lineWidth = 3; const x = crop[0] + (crop[2] * split) / 100; c.beginPath(); c.moveTo(x, crop[1]); c.lineTo(x, crop[1] + crop[3]); c.stroke() }
    }
    c.restore()
  }, [crop, imgB, mesh, Vedit, poseB, KB, alphaB, layer, polys, draftB, split])
  useEffect(() => {
    if (stage === 4 && !cropping && bRef.current && imgB && poseB && crop) renderB(bRef.current, { guides: true, cmp: compare })
  }, [stage, cropping, imgB, poseB, crop, renderB, compare])
  useEffect(() => {
    if (stage !== 4 || !cropping || !cropRef.current || !imgB) return
    const c = cropRef.current.getContext('2d')
    c.drawImage(imgB, 0, 0)
    if (crop) { c.strokeStyle = '#facc15'; c.lineWidth = 4; c.strokeRect(crop[0], crop[1], crop[2], crop[3]) }
  }, [stage, cropping, imgB, crop])

  const cropDown = (e) => { dragCrop.current = toImg(e, cropRef.current, photoB.width, photoB.height); cropRef.current.setPointerCapture(e.pointerId) }
  const cropMove = (e) => {
    if (!dragCrop.current) return
    const [x, y] = toImg(e, cropRef.current, photoB.width, photoB.height), [x0, y0] = dragCrop.current
    setCrop([Math.max(0, Math.min(x, x0)), Math.max(0, Math.min(y, y0)), Math.min(Math.abs(x - x0), photoB.width), Math.min(Math.abs(y - y0), photoB.height)].map((v) => Math.round(v)))
  }
  const cropUp = () => { dragCrop.current = null }
  const bDown = (e) => {
    if (!drawB) return
    const [x, y] = toImg(e, bRef.current, crop[2], crop[3])
    setDraftB([...draftB, [Math.round(x + crop[0]), Math.round(y + crop[1])]])
  }
  const savePlacement = () => guard('Saving…', async () => setSession(await v1.photoB(session.id, {
    crop, pose: poseToJson(poseB), fov_deg: fovB, opacity: alphaB, layer, mask_polygons: polys, version: session.versions.length })))
  const exportImg = (type) => {
    const cv = document.createElement('canvas'); cv.width = crop[2]; cv.height = crop[3]
    renderB(cv, { guides: false, cmp: false })
    cv.toBlob((b) => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = `smile_simulation.${type === 'png' ? 'png' : 'jpg'}`; a.click() }, `image/${type === 'png' ? 'png' : 'jpeg'}`, 0.92)
  }

  const nextOK = [true, !!session, !!reg, session?.stage === 'editing' || session?.stage === 'simulating', !!session?.alignment_a]
  const cancompare = !!poseB

  return (
    <>
      <header><h1>Smile simulation <span className="muted">v1 · research prototype</span></h1></header>
      <div className="banner">
        Automatic model selection is a geometric best guess from a template library; the 3D model is not the patient’s real anatomy and results are
        simulations, not clinical predictions. Do not use for diagnosis or treatment decisions.
      </div>
      <nav>{STAGES.map((s, i) => <button key={s} className={stage === i ? 'on' : ''} disabled={!nextOK[i]} onClick={() => setStage(i)}>{s}</button>)}</nav>
      <main>
        {busy && <div className="card muted">⏳ {busy}</div>}
        {err && <div className="card err" role="alert">⚠ {err}</div>}

        {stage === 0 && (
          <div className="card">
            <p className="muted">Catalogue: {catalogue.length} models loaded.</p>
            <div className="row">
              <div><h3>Photo A — frontal, retractor</h3><input type="file" accept=".png,.jpg,.jpeg" onChange={(e) => upload('A')(e.target.files[0])} />
                {photoA && <><p className="muted">{photoA.name} {photoA.width}×{photoA.height}</p><img src={photoA.url} alt="Photo A" style={{ maxWidth: 380 }} /></>}</div>
              <div><h3>Photo B — smiling face</h3><input type="file" accept=".png,.jpg,.jpeg" onChange={(e) => upload('B')(e.target.files[0])} />
                {photoB && <><p className="muted">{photoB.name} {photoB.width}×{photoB.height}</p><img src={photoB.url} alt="Photo B" style={{ maxWidth: 380 }} /></>}</div>
            </div>
            <p><button className="p" disabled={!photoA || !!busy} onClick={analyse}>Analyse Photo A →</button>{' '}
              <button className="s" onClick={demo} disabled={!!busy}>Load synthetic demo photos</button></p>
            <p className="muted">Photos are processed by the server for this session only and are not committed anywhere. Photo B is optional until step 5.</p>
          </div>
        )}

        {stage === 1 && session && photoA && (
          <div className="card">
            <div className="row">
              <div style={{ flex: '1 1 560px' }}>
                <canvas ref={annRef} width={photoA.width} height={photoA.height} style={{ width: '100%', maxWidth: 900, cursor: drawMode ? 'crosshair' : 'grab' }}
                  onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} />
                <p className="muted">Cyan = detected upper-tooth region (heuristic, editable). Yellow dots = canine reference points ({session.analysis.source || 'estimated'}); drag them onto the real canines.</p>
              </div>
              <div style={{ minWidth: 300, flex: 1 }}>
                <p><span className={`badge ${session.analysis.confidence === 'none' ? 'warn' : ''}`}>region: {session.analysis.confidence}</span>
                  {(session.analysis.notes || []).map((n) => <Fragment key={n}><br /><span className="muted">• {n}</span></Fragment>)}</p>
                <p><button className="s" onClick={() => { setDrawMode(!drawMode); setDraft([]) }}>{drawMode ? 'Cancel outline' : 'Redraw tooth outline'}</button>{' '}
                  {drawMode && <button className="s" disabled={draft.length < 3} onClick={closeDraft}>Close outline</button>}</p>
                <p><label>Approx. camera FOV (°) <input type="number" min="10" max="100" value={fov} onChange={(e) => setFov(+e.target.value)} style={{ width: 64 }} /></label>
                  <span className="muted"> (uncalibrated)</span></p>
                <p><button className="p" disabled={!!busy || !session.analysis.mask_polygons.length || !session.analysis.canines} onClick={runMatch}>Find best-fit model automatically</button></p>
                {session.match && (
                  <>
                    <p><b>Selected: {session.model_id.replace('lib:', '')}</b>{' '}
                      <span className={`badge ${session.match.preliminary ? 'warn' : ''}`}>{session.match.preliminary ? 'preliminary match' : 'match'}</span></p>
                    <p>{session.match.explanation}</p>
                    <table className="rank"><thead><tr><th>model</th><th>score</th><th>IoU</th><th>canine px</th></tr></thead><tbody>
                      {session.match.ranked.filter((r) => r.final !== undefined).slice(0, 6).map((r) => (
                        <tr key={r.id}><td>{r.id.replace('upper_', '')}</td><td>{f2(r.final)}</td><td>{f2(r.fit_iou)}</td><td>{f2(r.canine_rmse_px, 0)}</td></tr>))}
                    </tbody></table>
                    <p className="muted">{session.match.note} Shade is not used for matching.</p>
                    <p>Replace with: <select value="" onChange={(e) => replace(e.target.value)}>
                      <option value="">— choose —</option>
                      {catalogue.filter((m) => m.jaw === 'upper').map((m) => <option key={m.id} value={m.id}>{m.id}{m.classification !== 'complete_arch' ? ' (partial)' : ''}</option>)}
                    </select></p>
                    <button className="p" onClick={() => setStage(2)}>Align on Photo A →</button>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {stage === 2 && reg && mesh && photoA && poseA && (
          <div className="card">
            <div className="row">
              <div style={{ flex: '1 1 560px' }}>
                <div className="zoomwrap"><canvas ref={alignRef} width={photoA.width} height={photoA.height} style={{ width: `${zoom * 100}%`, display: 'block' }} /></div>
                <p className="muted">Yellow + = your canine points; orange ○ = the model’s canines. Two points do not fix depth or full 3D pose — adjust by eye.</p>
              </div>
              <div>
                <p><b>{session.model_id.replace('lib:', '')}</b> {session.match?.preliminary && <span className="badge warn">preliminary match</span>}
                  <br /><span className="muted">auto placement: {reg.report?.method || 'initial'} · {reg.report?.status || ''} · IoU {f2(reg.report?.metrics?.silhouette_iou)}
                    · {reg.camera.calibrated ? 'calibrated' : 'approximate'} camera</span></p>
                <div className="ctl">
                  <Slider label="Opacity" value={alpha} min={0} max={1} step={0.05} onChange={setAlpha} />
                  <Slider label="Zoom" value={zoom} min={1} max={4} step={0.1} onChange={setZoom} unit="×" />
                  <Slider label="Move X" value={d.t[0]} min={-40} max={40} step={0.1} unit=" mm" onStart={snap} onChange={(v) => setD({ ...d, t: [v, d.t[1], d.t[2]] })} />
                  <Slider label="Move Y" value={d.t[1]} min={-40} max={40} step={0.1} unit=" mm" onStart={snap} onChange={(v) => setD({ ...d, t: [d.t[0], v, d.t[2]] })} />
                  <Slider label="Depth Z" value={d.t[2]} min={-60} max={60} step={0.1} unit=" mm" onStart={snap} onChange={(v) => setD({ ...d, t: [d.t[0], d.t[1], v] })} />
                  <Slider label="Tilt (X)" value={d.rot[0]} min={-30} max={30} step={0.1} unit="°" onStart={snap} onChange={(v) => setD({ ...d, rot: [v, d.rot[1], d.rot[2]] })} />
                  <Slider label="Turn (Y)" value={d.rot[1]} min={-30} max={30} step={0.1} unit="°" onStart={snap} onChange={(v) => setD({ ...d, rot: [d.rot[0], v, d.rot[2]] })} />
                  <Slider label="Roll (Z)" value={d.rot[2]} min={-30} max={30} step={0.1} unit="°" onStart={snap} onChange={(v) => setD({ ...d, rot: [d.rot[0], d.rot[1], v] })} />
                  <Slider label="Scale" value={d.scale} min={0.8} max={1.25} step={0.005} unit="×" onStart={snap} onChange={(v) => setD({ ...d, scale: v })} />
                </div>
                <p><button className="s" onClick={undo} disabled={!hist.past.length}>Undo</button>{' '}
                  <button className="s" onClick={redo} disabled={!hist.future.length}>Redo</button>{' '}
                  <button className="s" onClick={() => { snap(); setD(ZERO()); setZoom(1) }}>Reset</button></p>
                <p><button className="p" disabled={!!busy} onClick={confirm}>Confirm alignment → open 3D editor</button></p>
              </div>
            </div>
          </div>
        )}

        {stage === 3 && mesh && session?.alignment_a && (
          <div className="card">
            <div className="row">
              <div style={{ flex: '1 1 560px' }}><Viewer meshes={viewMeshes} wireframe={false} selected={sel} />
                <p className="muted">Working copy of {session.model_id.replace('lib:', '')} (original library file is untouched). Drag = rotate, right-drag = pan, wheel = zoom.</p></div>
              <div style={{ minWidth: 300 }}>
                {!mesh.tooth_editing ? <p className="warn">{mesh.tooth_editing_note}</p> : (
                  <>
                    <p>Tooth: <select value={sel} onChange={(e) => setSel(e.target.value)}><option value="">— select —</option>{Object.keys(mesh.teeth).map((t) => <option key={t}>{t}</option>)}</select> (orange)</p>
                    {sel && (() => {
                      const o = offsets[sel] || { t: [0, 0, 0], r: [0, 0, 0] }
                      const set = (k, i, v) => { const n = { t: [...o.t], r: [...o.r] }; n[k][i] = v; setOffsets({ ...offsets, [sel]: n }) }
                      return <div className="ctl">
                        {['X', 'Y', 'Z'].map((a, i) => <Slider key={'t' + a} label={`Move ${a}`} value={o.t[i]} min={-5} max={5} step={0.05} unit=" mm" onChange={(v) => set('t', i, v)} />)}
                        {['X', 'Y', 'Z'].map((a, i) => <Slider key={'r' + a} label={`Rotate ${a}`} value={o.r[i]} min={-30} max={30} step={0.5} unit="°" onChange={(v) => set('r', i, v)} />)}
                        <button className="s" onClick={() => { const { [sel]: _, ...rest } = offsets; setOffsets(rest) }}>Reset tooth</button></div>
                    })()}
                    <button className="s" onClick={() => setOffsets({})}>Reset all teeth</button>
                  </>
                )}
                <p><input placeholder="version note" value={note} onChange={(e) => setNote(e.target.value)} />{' '}
                  <button className="p" disabled={!!busy} onClick={saveVersion}>Save version</button></p>
                <table><tbody>{session.versions.map((v) => (
                  <tr key={v.n}><td>v{v.n}</td><td>{v.note || '—'}</td><td className="muted">{Object.keys(v.tooth_offsets || {}).length} tooth edits</td>
                    <td><button className="s" onClick={() => restore(v.n)}>Restore</button></td></tr>))}</tbody></table>
                <p><button className="p" disabled={!photoB} onClick={() => { setStage(4); setCropping(true); if (!crop && photoB) setCrop([Math.round(photoB.width * 0.15), Math.round(photoB.height * 0.3), Math.round(photoB.width * 0.7), Math.round(photoB.height * 0.4)]) }}>
                  Open Photo B →</button>{!photoB && <span className="warn"> upload Photo B in step 1</span>}</p>
              </div>
            </div>
          </div>
        )}

        {stage === 4 && photoB && mesh && (
          <div className="card">
            {cropping ? (
              <>
                <p>Drag on Photo B to crop the mouth region.</p>
                <canvas ref={cropRef} width={photoB.width} height={photoB.height} style={{ width: '100%', maxWidth: 900, cursor: 'crosshair' }}
                  onPointerDown={cropDown} onPointerMove={cropMove} onPointerUp={cropUp} />
                <p><button className="p" disabled={!crop || crop[2] < 20 || crop[3] < 20} onClick={() => setCropping(false)}>Use this crop →</button></p>
              </>
            ) : poseB && (
              <div className="row">
                <div style={{ flex: '1 1 560px' }}>
                  <canvas ref={bRef} width={crop[2]} height={crop[3]} style={{ width: '100%', maxWidth: 900, cursor: drawB ? 'crosshair' : 'default' }} onPointerDown={bDown} />
                  <p className="muted">Photo B has its own camera and placement — independent of Photo A. Edit version used: current editor state.</p>
                </div>
                <div>
                  <div className="ctl">
                    <Slider label="Opacity" value={alphaB} min={0} max={1} step={0.05} onChange={setAlphaB} />
                    <Slider label="Field of view" value={fovB} min={15} max={90} step={1} unit="°" onChange={setFovB} />
                    <Slider label="Move X" value={dB.t[0]} min={-60} max={60} step={0.1} unit=" mm" onChange={(v) => setDB({ ...dB, t: [v, dB.t[1], dB.t[2]] })} />
                    <Slider label="Move Y" value={dB.t[1]} min={-60} max={60} step={0.1} unit=" mm" onChange={(v) => setDB({ ...dB, t: [dB.t[0], v, dB.t[2]] })} />
                    <Slider label="Depth Z" value={dB.t[2]} min={-150} max={150} step={0.5} unit=" mm" onChange={(v) => setDB({ ...dB, t: [dB.t[0], dB.t[1], v] })} />
                    <Slider label="Tilt (X)" value={dB.rot[0]} min={-30} max={30} step={0.1} unit="°" onChange={(v) => setDB({ ...dB, rot: [v, dB.rot[1], dB.rot[2]] })} />
                    <Slider label="Turn (Y)" value={dB.rot[1]} min={-30} max={30} step={0.1} unit="°" onChange={(v) => setDB({ ...dB, rot: [dB.rot[0], v, dB.rot[2]] })} />
                    <Slider label="Roll (Z)" value={dB.rot[2]} min={-30} max={30} step={0.1} unit="°" onChange={(v) => setDB({ ...dB, rot: [dB.rot[0], dB.rot[1], v] })} />
                    <Slider label="Scale" value={dB.scale} min={0.5} max={1.8} step={0.01} unit="×" onChange={(v) => setDB({ ...dB, scale: v })} />
                  </div>
                  <p>Layer order: <select value={layer} onChange={(e) => setLayer(e.target.value)}>
                    <option value="over">model on top (outlined regions hide it, e.g. lips)</option>
                    <option value="under">photo on top (model shows only inside outlined regions)</option></select></p>
                  <p><button className="s" onClick={() => { setDrawB(!drawB); setDraftB([]) }}>{drawB ? 'Stop drawing' : 'Draw visible-region mask'}</button>{' '}
                    {drawB && <button className="s" disabled={draftB.length < 3} onClick={() => { setPolys([...polys, draftB]); setDraftB([]) }}>Close region</button>}{' '}
                    <button className="s" disabled={!polys.length} onClick={() => setPolys([])}>Clear masks ({polys.length})</button></p>
                  <p><label><input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} /> before/after</label>
                    {compare && <input type="range" min="0" max="100" value={split} onChange={(e) => setSplit(+e.target.value)} />}</p>
                  <p><button className="s" onClick={() => setDB(ZERO())}>Reset placement</button>{' '}<button className="s" onClick={() => setCropping(true)}>Re-crop</button></p>
                  <p><button className="p" onClick={() => exportImg('png')}>Export PNG</button>{' '}<button className="s" onClick={() => exportImg('jpeg')}>Export JPEG</button>{' '}
                    <button className="s" onClick={savePlacement} disabled={!!busy}>Save placement</button></p>
                  <p className="muted">Exported image shows the simulation only (no guides). Placement: {session.photo_b_placement ? 'saved' : 'not saved yet'}.</p>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </>
  )
}
