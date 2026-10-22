async function call(path, opts = {}) {
  const r = await fetch(path, opts)
  const j = await r.json().catch(() => ({ ok: false, error: `HTTP ${r.status}` }))
  if (!r.ok || j.ok === false) throw new Error(j.error || `HTTP ${r.status}`)
  return j
}
const json = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) })
const form = (file) => { const f = new FormData(); f.append('file', file); return { method: 'POST', body: f } }

export const api = {
  models: () => call('/api/models'),
  uploadModel: (file) => call('/api/models', form(file)),
  mesh: (id, faces = 12000) => call(`/api/models/${id.replace(/^(\w+):/, '$1/')}/mesh?target_faces=${faces}`),
  uploadImage: (file) => call('/api/images', form(file)),
  createReg: (body) => call('/api/registrations', json('POST', body)),
  setAnnotations: (id, ann) => call(`/api/registrations/${id}/annotations`, json('PUT', ann)),
  auto: (id, config) => call(`/api/registrations/${id}/auto`, json('POST', { config })),
  transform: (id, body) => call(`/api/registrations/${id}/transform`, json('POST', body)),
  exportReg: (id) => call(`/api/registrations/${id}/export`, json('POST')),
  load: (alignment, image_id) => call('/api/registrations/load', json('POST', { alignment, image_id })),
  sampleBlob: async (name) => { const r = await fetch(`/api/samples/${name}`); if (!r.ok) throw new Error('sample not available'); return r.blob() },
  exportUrl: (id, name) => `/api/registrations/${id}/export/${name}`,
  imageUrl: (id) => `/api/images/${id}/file`,
}
