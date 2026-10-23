// API client for the v1 demo.
//   VITE_API_BASE  base URL of the backend ('' = same origin; set to the backend's URL when hosted separately, e.g. Vercel)
//   VITE_API_MODE  'stateful' (default: server keeps sessions - local/Docker/Render) or 'stateless' (serverless: the browser
//                  keeps the session; the server only computes - no durable storage needed)
const BASE = import.meta.env.VITE_API_BASE || 'https://vita-3d-master-shade-system-for-den.vercel.app'
export const STATELESS = import.meta.env.VITE_API_MODE === 'stateless'
const MAX_UPLOAD_DIM = 1600 // px; keeps Photo A well under Vercel's 4.5 MB request body limit
const MAX_UPLOAD_BYTES = 3.5 * 2 ** 20

async function call(path, opts = {}) {
  let r
  try { r = await fetch(BASE + path, opts) } catch (e) { throw new Error(`Cannot reach the API (${BASE || 'same origin'}): ${e.message}`) }
  const j = await r.json().catch(() => ({ ok: false, error: r.status === 413 ? 'Upload too large for the server (max ~4.5 MB)' : `HTTP ${r.status}` }))
  if (!r.ok || j.ok === false) throw new Error(j.error || (j.detail && JSON.stringify(j.detail)) || `HTTP ${r.status}`)
  return j
}
const json = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) })
const form = (file) => { const f = new FormData(); f.append('file', file); return { method: 'POST', body: f } }
const common = {
  health: () => call('/api/health'),
  catalogue: () => call('/api/v1/catalogue'),
  sample: async (name) => { const r = await fetch(`${BASE}/api/samples/${name}`); if (!r.ok) throw new Error('demo sample not available'); return r.blob() },
  mesh: (modelId, faces = 12000) => call(`/api/models/${modelId.replace(':', '/')}/mesh?target_faces=${faces}`),
}

// ------------------------------------------------------------------ stateful (server keeps sessions)
const stateful = {
  ...common,
  uploadImage: async (file) => ({ ...(await call('/api/images', form(file))), blob: file }),
  createSession: (a, b) => call('/api/v1/sessions', json('POST', { photo_a_id: a, photo_b_id: b })),
  setAnnotations: (id, body) => call(`/api/v1/sessions/${id}/annotations`, json('PUT', body)),
  match: (id, body) => call(`/api/v1/sessions/${id}/match`, json('POST', body)),
  replace: (id, model_id) => call(`/api/v1/sessions/${id}/replace_model`, json('POST', { model_id })),
  confirm: (id, pose) => call(`/api/v1/sessions/${id}/confirm_alignment`, json('POST', pose ? { pose } : {})),
  saveVersion: (id, tooth_offsets, note) => call(`/api/v1/sessions/${id}/versions`, json('POST', { tooth_offsets, note })),
  restore: (id, n) => call(`/api/v1/sessions/${id}/versions/${n}/restore`, json('POST')),
  photoB: (id, body) => call(`/api/v1/sessions/${id}/photo_b`, json('PUT', body)),
}

// ------------------------------------------------------------------ stateless (browser keeps the session)
const files = new Map() // image id -> {blob,width,height}
const sessions = new Map() // session id -> session object (same shape the stateful API returns)
const now = () => new Date().toISOString().replace(/\.\d+Z$/, 'Z')
const rid = () => Math.random().toString(36).slice(2, 14)

async function decode(file) {
  const bmp = await createImageBitmap(file).catch(() => { throw new Error('Photograph could not be decoded; use PNG or JPEG') })
  return bmp
}
async function shrink(file) { // Photo A only: downscale so the upload fits the platform body limit
  const bmp = await decode(file)
  const s = Math.min(1, MAX_UPLOAD_DIM / Math.max(bmp.width, bmp.height))
  if (s === 1 && file.size <= MAX_UPLOAD_BYTES) return { blob: file, width: bmp.width, height: bmp.height }
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s)
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h
  cv.getContext('2d').drawImage(bmp, 0, 0, w, h)
  const blob = await new Promise((res) => cv.toBlob(res, 'image/jpeg', 0.92))
  return { blob, width: w, height: h }
}
const need = (id) => { const s = sessions.get(id); if (!s) throw new Error('Session expired - start again from the upload step'); return s }
const save = (s) => { sessions.set(s.id, s); return { ok: true, ...s } }

const stateless = {
  ...common,
  uploadImage: async (file, role = 'A') => {
    if (!/\.(png|jpe?g)$/i.test(file.name)) throw new Error('Unsupported image format; use PNG or JPEG')
    const info = role === 'A' ? await shrink(file) : await decode(file).then((b) => ({ blob: file, width: b.width, height: b.height }))
    const id = rid(); files.set(id, info)
    return { ok: true, id, width: info.width, height: info.height, name: file.name, blob: info.blob }
  },
  createSession: async (a, b) => {
    const A = files.get(a)
    if (!A) throw new Error('Upload Photo A first')
    const r = await call('/api/v1/stateless/analyse', form(new File([A.blob], 'photo_a.jpg')))
    const B = b && files.get(b)
    return save({ id: rid(), created: now(), photo_a: { id: a, width: r.width, height: r.height }, photo_b: B ? { id: b, width: B.width, height: B.height } : null,
      analysis: r.analysis, match: null, registration_id: null, model_id: null, alignment_a: null, versions: [], photo_b_placement: null, stage: 'uploaded' })
  },
  setAnnotations: async (id, body) => {
    const s = need(id), { width: w, height: h } = s.photo_a
    const inside = (p) => p[0] >= 0 && p[0] <= w - 1 && p[1] >= 0 && p[1] <= h - 1
    if (body.canines) {
      if (body.canines.length !== 2 || !body.canines.every(inside)) throw new Error('canines: exactly two points inside the image are required')
      s.analysis.canines = body.canines
    }
    if (body.mask_polygons) {
      if (!body.mask_polygons.every((poly) => poly.length >= 3 && poly.every(inside))) throw new Error('mask polygon: >=3 points inside the image are required')
      s.analysis.mask_polygons = body.mask_polygons
    }
    s.analysis.source = 'user-edited'
    return save(s)
  },
  match: async (id, body = {}) => {
    const s = need(id), a = s.analysis
    const r = await call('/api/v1/stateless/match', json('POST', { width: s.photo_a.width, height: s.photo_a.height, canines: a.canines,
      mask_polygons: a.mask_polygons, fov_deg: body.fov_deg ?? 40, top_k: body.top_k ?? 6 }))
    Object.assign(s, { match: r.match, model_id: r.model_id, fov_deg: r.fov_deg, registration_id: r.registration.id, registration: r.registration, stage: 'matched' })
    return { ...save(s), registration: r.registration }
  },
  replace: async (id, model_id) => {
    const s = need(id), a = s.analysis
    const r = await call('/api/v1/stateless/place', json('POST', { width: s.photo_a.width, height: s.photo_a.height, canines: a.canines,
      mask_polygons: a.mask_polygons, fov_deg: s.fov_deg ?? 40, model_id }))
    if (s.match) s.match.user_replaced = model_id
    Object.assign(s, { model_id: r.model_id, registration_id: r.registration.id, registration: r.registration, stage: 'matched' })
    return { ...save(s), registration: r.registration }
  },
  confirm: async (id, pose) => {
    const s = need(id)
    let reg = s.registration
    if (!reg) throw new Error('Run model matching first')
    if (pose) reg = (await call('/api/v1/stateless/transform', json('POST', { registration: reg, pose }))).registration
    s.registration = reg
    s.alignment_a = { pose: reg.pose, camera: reg.camera, model_id: s.model_id, report: reg.report }
    s.versions = [{ n: 1, note: 'confirmed alignment (unedited working copy)', tooth_offsets: {}, time: now() }]
    s.stage = 'editing'
    return save(s)
  },
  saveVersion: async (id, tooth_offsets, note = '') => {
    const s = need(id)
    s.versions.push({ n: s.versions.length + 1, note: String(note).slice(0, 200), tooth_offsets: tooth_offsets || {}, time: now() })
    return save(s)
  },
  restore: async (id, n) => {
    const s = need(id), v = s.versions.find((x) => x.n === n)
    if (!v) throw new Error(`version ${n} not found`)
    s.versions.push({ ...v, n: s.versions.length + 1, note: `restored from v${n}` })
    return save(s)
  },
  photoB: async (id, body) => {
    const s = need(id)
    if (!s.photo_b) throw new Error('Upload Photo B first')
    s.photo_b_placement = Object.fromEntries(['crop', 'pose', 'fov_deg', 'opacity', 'layer', 'mask_polygons', 'version'].map((k) => [k, body[k] ?? null]))
    s.stage = 'simulating'
    return save(s)
  },
}

export const v1 = STATELESS ? stateless : stateful
