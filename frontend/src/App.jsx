import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api.js'
import Viewer from './Viewer.jsx'
import { drawMesh } from './draw.js'
import { applyToothOffsets, composePose, poseFromJson, poseToJson } from './geom.js'

const STEPS = ['1 Input', '2 Markers & boundaries', '3 3D viewer', '4 Alignment', '5 Tooth editing', '6 Results']
const ZERO = () => ({ rot: [0, 0, 0], t: [0, 0, 0], scale: 1 })
const fmt = (x, n = 3) => (typeof x === 'number' ? x.toFixed(n) : '—')

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

function AlignCanvas({ photo, img, mesh, V, pose, K, alpha, selected, side, wire }) {
  const a = useRef(), b = useRef()
  useEffect(() => {
    if (!img || !mesh || !a.current) return
    const w = photo.width, h = photo.height
    const c1 = a.current.getContext('2d')
    c1.clearRect(0, 0, w, h); c1.drawImage(img, 0, 0, w, h)
    let ctx = c1
    if (side && b.current) {
      ctx = b.current.getContext('2d')
      ctx.fillStyle = '#1d2024'; ctx.fillRect(0, 0, w, h)
    }
    drawMesh(ctx, mesh, V, pose, K, { alpha: side ? 1 : alpha, selected, wire })
  }, [img, mesh, V, pose, K, alpha, selected, side, wire, photo])
  return (
    <div className="row">
      <canvas ref={a} width={photo.width} height={photo.height} style={{ width: side ? '48%' : '100%', maxWidth: 900, height: 'auto' }} />
      {side && <canvas ref={b} width={photo.width} height={photo.height} style={{ width: '48%', maxWidth: 900, height: 'auto' }} />}
    </div>
  )
}

function Slider({ label, value, min, max, step, onChange, unit = '' }) {
  return (
    <>
      <label>{label}</label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
      <span className="muted">{fmt(value, 2)}{unit}</span>
    </>
  )
}

export default function App() {
  const [step, setStep] = useState(0)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const [photo, setPhoto] = useState(null) // {id,width,height,name,url}
  const [models, setModels] = useState([])
  const [modelId, setModelId] = useState('')
  const [lowerId, setLowerId] = useState('')
  const [mesh, setMesh] = useState(null)
  const [lowerMesh, setLowerMesh] = useState(null)
  const [fov, setFov] = useState(40)
  const [reg, setReg] = useState(null)
  const [ann, setAnn] = useState({ markers: [], landmarks: [], mask_polygons: [] })
  const [base, setBase] = useState(null)
  const [d, setD] = useState(ZERO())
  const [offsets, setOffsets] = useState({})
  const [alpha, setAlpha] = useState(0.55)
  const [side, setSide] = useState(false)
  const [wire, setWire] = useState(false)
  const [freeScale, setFreeScale] = useState(false)
  const [selTooth, setSelTooth] = useState('')
  const [exported, setExported] = useState(null)
  const img = useImage(photo?.url)

  const guard = useCallback(async (fn) => {
    setErr(''); setBusy(true)
    try { return await fn() } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }, [])

  useEffect(() => { api.models().then((r) => setModels(r.models)).catch((e) => setErr(e.message)) }, [])
  useEffect(() => {
    setMesh(null); setReg(null); setExported(null)
    if (modelId) guard(async () => setMesh(await api.mesh(modelId)))
  }, [modelId])
  useEffect(() => {
    setLowerMesh(null)
    if (lowerId) guard(async () => setLowerMesh(await api.mesh(lowerId)))
  }, [lowerId])

  const K = reg?.camera.K
  const pose = useMemo(() => (base ? composePose(base, d) : null), [base, d])
  const V = useMemo(() => (mesh ? applyToothOffsets(Float32Array.from(mesh.vertices), mesh.faces, mesh.face_tooth, mesh.teeth, offsets) : null), [mesh, offsets])

  const adopt = (r) => { setReg(r); setBase(poseFromJson(r.pose)); setD(ZERO()); setOffsets(r.tooth_offsets || {}) }

  const ready = photo && mesh
  const ensureReg = () => guard(async () => {
    if (reg || !ready) return
    adopt(await api.createReg({ image_id: photo.id, model_id: modelId, camera: { fov_deg: fov }, annotations: ann }))
  })
  useEffect(() => { if (step >= 1 && ready && !reg) ensureReg() }, [step, ready, reg])

  // -------- step 1 handlers
  const onPhoto = (f) => f && guard(async () => {
    const r = await api.uploadImage(f)
    setPhoto({ ...r, url: URL.createObjectURL(f) }); setReg(null)
  })
  const onModel = (f) => f && guard(async () => {
    const r = await api.uploadModel(f)
    setModels((await api.models()).models); setModelId(r.id)
  })
  const loadSample = () => guard(async () => {
    const blob = await api.sampleBlob('synthetic_photo.png')
    const r = await api.uploadImage(new File([blob], 'synthetic_photo.png', { type: 'image/png' }))
    setPhoto({ ...r, url: URL.createObjectURL(blob) })
    const a = await (await api.sampleBlob('synthetic_annotations.json')).text()
    const { _note, ...rest } = JSON.parse(a)
    setAnn(rest)
    const up = models.find((m) => m.id.endsWith('upper_01346914'))
    if (up) setModelId(up.id)
    setReg(null)
  })

  // -------- step 2: annotation canvas
  const [tool, setTool] = useState('landmark')
  const [lmTooth, setLmTooth] = useState('11')
  const [draft, setDraft] = useState([])
  const [show, setShow] = useState({ markers: true, landmarks: true, mask: true })
  const annCanvas = useRef()
  useEffect(() => {
    if (step !== 1 || !img || !annCanvas.current) return
    const c = annCanvas.current.getContext('2d')
    c.drawImage(img, 0, 0)
    const dot = (p, col, label) => {
      c.fillStyle = col; c.beginPath(); c.arc(p[0], p[1], 6, 0, 7); c.fill()
      c.font = '16px sans-serif'; c.fillStyle = '#fff'; c.strokeStyle = '#000'; c.lineWidth = 3
      c.strokeText(label, p[0] + 9, p[1] - 6); c.fillText(label, p[0] + 9, p[1] - 6)
    }
    const poly = (pts, col, closed) => {
      c.strokeStyle = col; c.lineWidth = 3; c.beginPath()
      pts.forEach((p, i) => (i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])))
      if (closed) c.closePath()
      c.stroke()
    }
    if (show.mask) { ann.mask_polygons.forEach((p) => poly(p, '#22d3ee', true)); poly(draft, '#fde047', false) }
    if (show.markers) ann.markers.forEach((m) => dot(m.image_xy, '#f43f5e', m.id))
    if (show.landmarks) ann.landmarks.forEach((m) => dot(m.image_xy, '#a3e635', String(m.fdi)))
  }, [step, img, ann, draft, show])
  const onAnnClick = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const p = [Math.round(((e.clientX - r.left) * photo.width) / r.width), Math.round(((e.clientY - r.top) * photo.height) / r.height)]
    if (tool === 'marker') setAnn({ ...ann, markers: [...ann.markers, { id: `M${ann.markers.length + 1}`, image_xy: p }] })
    else if (tool === 'landmark')
      setAnn({ ...ann, landmarks: [...ann.landmarks.filter((l) => String(l.fdi) !== lmTooth), { fdi: +lmTooth, image_xy: p }] })
    else setDraft([...draft, p])
  }
  const saveAnn = () => guard(async () => reg && setReg(await api.setAnnotations(reg.id, ann)))

  // -------- step 4 handlers
  const runAuto = () => guard(async () => {
    await api.setAnnotations(reg.id, ann)
    adopt(await api.auto(reg.id, { fix_scale: !freeScale }))
  })
  const dirty = d.scale !== 1 || d.rot.some(Boolean) || d.t.some(Boolean) || JSON.stringify(offsets) !== JSON.stringify(reg?.tooth_offsets || {})
  const sync = async () => {
    if (!dirty) return // unchanged: keep the automatic result's method/status/metrics
    const r = await api.transform(reg.id, { pose: poseToJson(pose), tooth_offsets: offsets })
    setReg(r); setBase(poseFromJson(r.pose)); setD(ZERO())
  }
  const save = () => guard(async () => { await sync(); setExported(null) })
  const doExport = () => guard(async () => { await sync(); setExported(await api.exportReg(reg.id)) })
  const loadAlignment = (f) => f && guard(async () => {
    const doc = JSON.parse(await f.text())
    if (!photo) throw new Error('Upload the photograph first (Input step), then load the alignment.')
    adopt(await api.load(doc, photo.id))
    setModelId(doc.model.id)
  })

  const hasTeeth = mesh?.tooth_editing
  const disabled = [false, !ready, !mesh, !reg, !reg, !reg]
  const ok = !!(reg && mesh && img && K)

  return (
    <>
      <header><h1>Dental photograph ↔ 3D model alignment <span className="muted">(research prototype)</span></h1></header>
      <div className="banner">
        The 3D model is a template/prior fitted to the photograph. A single photo cannot determine a patient’s true tooth geometry — results are
        estimates, not clinical measurements, and not a patient-specific reconstruction.
      </div>
      <nav>{STEPS.map((s, i) => <button key={s} className={step === i ? 'on' : ''} disabled={disabled[i]} onClick={() => setStep(i)}>{s}</button>)}</nav>
      <main>
        {err && <div className="card err" role="alert">⚠ {err}</div>}

        {step === 0 && (
          <div className="card">
            <h3>Photograph</h3>
            <input type="file" accept=".png,.jpg,.jpeg" onChange={(e) => onPhoto(e.target.files[0])} />{' '}
            <button className="s" onClick={loadSample} disabled={busy || !models.length}>Load synthetic demo (photo + annotations + upper model)</button>
            {photo && <p className="muted">{photo.name} — {photo.width}×{photo.height}px</p>}
            {photo && <img src={photo.url} alt="uploaded" style={{ maxWidth: 520, border: '1px solid var(--line)' }} />}
            <h3>3D dental model</h3>
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              <option value="">— select model —</option>
              {models.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.jaw}, {m.format})</option>)}
            </select>{' '}
            <input type="file" accept=".obj,.ply,.stl" onChange={(e) => onModel(e.target.files[0])} />
            {modelId && <p className="muted">Selected: <b>{models.find((m) => m.id === modelId)?.name}</b> · {models.find((m) => m.id === modelId)?.format} · units mm
              {mesh && !hasTeeth && <span className="warn"> · single mesh: tooth editing unavailable</span>}</p>}
            <h3>Camera</h3>
            <label>Approximate horizontal field of view (°): <input type="number" min="5" max="120" value={fov} onChange={(e) => { setFov(+e.target.value); setReg(null) }} style={{ width: 70 }} /></label>
            <p className="muted">No camera calibration is available, so an approximate pinhole camera is used. Calibrated intrinsics can be supplied via the API (<code>camera.K</code>).</p>
            <p>{ready ? <button className="p" onClick={() => setStep(1)}>Next →</button> : <span className="muted">Select a photograph and a model to continue.</span>}</p>
          </div>
        )}

        {step === 1 && photo && (
          <div className="card">
            <div className="row">
              <div>
                <div className="stage">
                  <canvas ref={annCanvas} width={photo.width} height={photo.height} onClick={onAnnClick} style={{ cursor: 'crosshair', width: '100%', maxWidth: 900 }} />
                </div>
              </div>
              <div style={{ minWidth: 280 }}>
                <p>Tool:{' '}
                  <select value={tool} onChange={(e) => setTool(e.target.value)}>
                    <option value="landmark">Tooth landmark (tooth centre)</option>
                    <option value="marker">Reference marker</option>
                    <option value="polygon">Tooth region outline</option>
                  </select>
                </p>
                {tool === 'landmark' && (
                  <p>Tooth (FDI):{' '}
                    <select value={lmTooth} onChange={(e) => setLmTooth(e.target.value)}>
                      {Object.keys(mesh?.teeth || {}).map((t) => <option key={t}>{t}</option>)}
                    </select>
                    <span className="muted"> click the visual centre of that tooth. ≥4 landmarks enable pose-from-correspondence (PnP).</span>
                  </p>
                )}
                {tool === 'polygon' && (
                  <p>{draft.length} points <button className="s" disabled={draft.length < 3} onClick={() => { setAnn({ ...ann, mask_polygons: [...ann.mask_polygons, draft] }); setDraft([]) }}>Close outline</button>{' '}
                    <button className="s" onClick={() => setDraft([])}>Clear draft</button></p>
                )}
                <p>
                  {['markers', 'landmarks', 'mask'].map((k) => <label key={k} style={{ marginRight: 10 }}><input type="checkbox" checked={show[k]} onChange={(e) => setShow({ ...show, [k]: e.target.checked })} /> {k}</label>)}
                </p>
                <p className="muted">Markers are recorded for scale/reference but are not used for pose unless their physical 3D arrangement is supplied. Tooth outlines are drawn manually here (no automatic segmentation is bundled).</p>
                <table><tbody>
                  {ann.markers.map((m, i) => <tr key={m.id}><td>marker {m.id}</td><td>{m.image_xy.join(', ')}</td><td><button className="s" onClick={() => setAnn({ ...ann, markers: ann.markers.filter((_, j) => j !== i) })}>×</button></td></tr>)}
                  {ann.landmarks.map((m, i) => <tr key={m.fdi}><td>tooth {m.fdi}</td><td>{m.image_xy.join(', ')}</td><td><button className="s" onClick={() => setAnn({ ...ann, landmarks: ann.landmarks.filter((_, j) => j !== i) })}>×</button></td></tr>)}
                  {ann.mask_polygons.map((p, i) => <tr key={i}><td>outline {i + 1}</td><td>{p.length} pts</td><td><button className="s" onClick={() => setAnn({ ...ann, mask_polygons: ann.mask_polygons.filter((_, j) => j !== i) })}>×</button></td></tr>)}
                </tbody></table>
                <p><button className="p" onClick={saveAnn} disabled={busy || !reg}>Send annotations to backend</button>{' '}
                  <button className="s" onClick={() => setStep(2)}>Next →</button></p>
              </div>
            </div>
          </div>
        )}

        {step === 2 && mesh && (
          <div className="card">
            <p>
              <label><input type="checkbox" checked={wire} onChange={(e) => setWire(e.target.checked)} /> wireframe</label>{' '}
              Second arch (display only):{' '}
              <select value={lowerId} onChange={(e) => setLowerId(e.target.value)}>
                <option value="">none</option>
                {models.filter((m) => m.id !== modelId).map((m) => <option key={m.id} value={m.id}>{m.name} ({m.jaw})</option>)}
              </select>
            </p>
            <Viewer meshes={[mesh, ...(lowerMesh ? [lowerMesh] : [])]} wireframe={wire} selected={null} />
            <p className="muted">Arches are shown side by side; no occlusal relationship between them is implied. Drag to rotate, right-drag to pan, wheel to zoom.</p>
            <button className="p" onClick={() => setStep(3)}>Next →</button>
          </div>
        )}

        {step === 3 && ok && pose && (
          <div className="card">
            <div className="row">
              <div style={{ flex: '1 1 520px' }}>
                <AlignCanvas photo={photo} img={img} mesh={mesh} V={V} pose={pose} K={K} alpha={alpha} selected={null} side={side} wire={wire} />
              </div>
              <div>
                <div className="ctl">
                  <Slider label="Opacity" value={alpha} min={0} max={1} step={0.05} onChange={setAlpha} />
                  <Slider label="Move X" value={d.t[0]} min={-40} max={40} step={0.1} unit=" mm" onChange={(v) => setD({ ...d, t: [v, d.t[1], d.t[2]] })} />
                  <Slider label="Move Y" value={d.t[1]} min={-40} max={40} step={0.1} unit=" mm" onChange={(v) => setD({ ...d, t: [d.t[0], v, d.t[2]] })} />
                  <Slider label="Depth Z" value={d.t[2]} min={-60} max={60} step={0.1} unit=" mm" onChange={(v) => setD({ ...d, t: [d.t[0], d.t[1], v] })} />
                  <Slider label="Tilt (X)" value={d.rot[0]} min={-30} max={30} step={0.1} unit="°" onChange={(v) => setD({ ...d, rot: [v, d.rot[1], d.rot[2]] })} />
                  <Slider label="Turn (Y)" value={d.rot[1]} min={-30} max={30} step={0.1} unit="°" onChange={(v) => setD({ ...d, rot: [d.rot[0], v, d.rot[2]] })} />
                  <Slider label="Roll (Z)" value={d.rot[2]} min={-30} max={30} step={0.1} unit="°" onChange={(v) => setD({ ...d, rot: [d.rot[0], d.rot[1], v] })} />
                  <Slider label="Scale" value={d.scale} min={0.8} max={1.25} step={0.005} unit="×" onChange={(v) => setD({ ...d, scale: v })} />
                </div>
                <p>
                  <label><input type="checkbox" checked={side} onChange={(e) => setSide(e.target.checked)} /> side-by-side</label>{' '}
                  <label><input type="checkbox" checked={wire} onChange={(e) => setWire(e.target.checked)} /> wireframe</label>
                </p>
                <p>
                  <button className="s" onClick={() => { setD(ZERO()); setOffsets({}) }}>Reset adjustments</button>{' '}
                  <button className="s" onClick={() => guard(async () => adopt(await api.createReg({ image_id: photo.id, model_id: modelId, camera: { fov_deg: fov }, annotations: ann })))}>Reset to initial pose</button>
                </p>
                <p>
                  <button className="p" onClick={runAuto} disabled={busy}>Run automatic alignment</button>{' '}
                  <label><input type="checkbox" checked={freeScale} onChange={(e) => setFreeScale(e.target.checked)} /> allow scale change</label>
                </p>
                <p className="muted">Automatic mode needs ≥4 tooth landmarks (PnP) or a tooth-region outline (silhouette fit); with neither, align manually.</p>
                <p><button className="s" onClick={save} disabled={busy}>Save alignment</button>{' '}<button className="s" onClick={() => setStep(5)}>Results →</button></p>
                {reg.report && <p className={reg.report.status === 'ok' || reg.report.status === 'manual' ? '' : 'warn'}>
                  <b>{reg.report.method}</b> · {reg.report.status}{(reg.report.notes || []).map((n) => <Fragment key={n}><br />• {n}</Fragment>)}</p>}
              </div>
            </div>
          </div>
        )}

        {step === 4 && ok && pose && (
          <div className="card">
            {!hasTeeth ? (
              <p className="warn">Individual tooth editing is disabled: the selected model is a single mesh, so teeth cannot be moved independently.</p>
            ) : (
              <div className="row">
                <div style={{ flex: '1 1 520px' }}>
                  <AlignCanvas photo={photo} img={img} mesh={mesh} V={V} pose={pose} K={K} alpha={alpha} selected={selTooth} side={false} wire={wire} />
                </div>
                <div>
                  <p>Tooth:{' '}
                    <select value={selTooth} onChange={(e) => setSelTooth(e.target.value)}>
                      <option value="">— select —</option>
                      {Object.keys(mesh.teeth).map((t) => <option key={t}>{t}</option>)}
                    </select>{' '}(highlighted in orange)</p>
                  {selTooth && (() => {
                    const o = offsets[selTooth] || { t: [0, 0, 0], r: [0, 0, 0] }
                    const set = (k, i, v) => { const n = { t: [...o.t], r: [...o.r] }; n[k][i] = v; setOffsets({ ...offsets, [selTooth]: n }) }
                    return (
                      <div className="ctl">
                        {['X', 'Y', 'Z'].map((a, i) => <Slider key={'t' + a} label={`Move ${a}`} value={o.t[i]} min={-5} max={5} step={0.05} unit=" mm" onChange={(v) => set('t', i, v)} />)}
                        {['X', 'Y', 'Z'].map((a, i) => <Slider key={'r' + a} label={`Rotate ${a}`} value={o.r[i]} min={-30} max={30} step={0.5} unit="°" onChange={(v) => set('r', i, v)} />)}
                        <button className="s" onClick={() => { const { [selTooth]: _, ...rest } = offsets; setOffsets(rest) }}>Reset tooth</button>
                      </div>
                    )
                  })()}
                  <p className="muted">Offsets are in model axes (mm / degrees about the tooth centre) and are exported with the alignment. Neighbouring teeth are not pushed aside.</p>
                  <button className="s" onClick={() => setOffsets({})}>Reset all teeth</button>
                </div>
              </div>
            )}
          </div>
        )}

        {step === 5 && ok && pose && (
          <div className="card">
            <p><b>Status:</b> {reg.status} · mode <b>{reg.mode}</b>{' '}
              {!reg.camera.calibrated && <span className="warn">· approximate camera (FOV {fmt(reg.camera.fov_deg, 1)}°, not calibrated)</span>}
              {reg.mode !== 'auto' && <span className="warn"> · manually adjusted — approximate</span>}</p>
            <div className="row">
              <div style={{ flex: '1 1 560px' }}>
                <AlignCanvas photo={photo} img={img} mesh={mesh} V={V} pose={pose} K={K} alpha={alpha} selected={null} side={true} wire={wire} />
                <p className="muted">Left: original photograph. Right: aligned model. Switch to overlay in the Alignment step.</p>
              </div>
              <div>
                <h4>Available metrics</h4>
                <table><tbody>
                  <tr><td>Landmark RMSE (px)</td><td>{fmt(reg.report?.metrics?.landmark_rmse_px, 2)}</td></tr>
                  <tr><td>Silhouette IoU</td><td>{fmt(reg.report?.metrics?.silhouette_iou, 3)}</td></tr>
                </tbody></table>
                <p className="muted">“—” means that metric cannot be computed from the annotations provided. Metrics measure agreement with <i>your annotations</i>, not anatomical accuracy.</p>
                <p><button className="p" onClick={doExport} disabled={busy}>Export</button>{' '}
                  <label className="s" style={{ cursor: 'pointer' }}>Load alignment JSON <input hidden type="file" accept=".json" onChange={(e) => loadAlignment(e.target.files[0])} /></label></p>
                {exported && (
                  <p>
                    {exported.files.map((f) => <span key={f}><a href={api.exportUrl(reg.id, f)} download>{f}</a>{' · '}</span>)}
                    <br /><span className="muted">Aligned meshes are in camera coordinates (mm). Original model files are never modified.</span>
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </>
  )
}
