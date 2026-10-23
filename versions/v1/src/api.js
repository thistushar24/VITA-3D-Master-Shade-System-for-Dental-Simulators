// API base: same origin by default (Render single service); override with VITE_API_BASE for a separate backend.
const BASE = import.meta.env.VITE_API_BASE || ''

async function call(path, opts = {}) {
  const r = await fetch(BASE + path, opts)
  const j = await r.json().catch(() => ({ ok: false, error: `HTTP ${r.status}` }))
  if (!r.ok || j.ok === false) throw new Error(j.error || `HTTP ${r.status}`)
  return j
}
const json = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) })
const form = (file) => { const f = new FormData(); f.append('file', file); return { method: 'POST', body: f } }

export const v1 = {
  health: () => call('/api/health'),
  catalogue: () => call('/api/v1/catalogue'),
  uploadImage: (file) => call('/api/images', form(file)),
  sample: async (name) => { const r = await fetch(`${BASE}/api/samples/${name}`); if (!r.ok) throw new Error('demo sample not available'); return r.blob() },
  createSession: (a, b) => call('/api/v1/sessions', json('POST', { photo_a_id: a, photo_b_id: b })),
  setAnnotations: (id, body) => call(`/api/v1/sessions/${id}/annotations`, json('PUT', body)),
  match: (id, body) => call(`/api/v1/sessions/${id}/match`, json('POST', body)),
  replace: (id, model_id) => call(`/api/v1/sessions/${id}/replace_model`, json('POST', { model_id })),
  confirm: (id, pose) => call(`/api/v1/sessions/${id}/confirm_alignment`, json('POST', pose ? { pose } : {})),
  saveVersion: (id, tooth_offsets, note) => call(`/api/v1/sessions/${id}/versions`, json('POST', { tooth_offsets, note })),
  restore: (id, n) => call(`/api/v1/sessions/${id}/versions/${n}/restore`, json('POST')),
  photoB: (id, body) => call(`/api/v1/sessions/${id}/photo_b`, json('PUT', body)),
  mesh: (modelId, faces = 12000) => call(`/api/models/${modelId.replace(':', '/')}/mesh?target_faces=${faces}`),
}
