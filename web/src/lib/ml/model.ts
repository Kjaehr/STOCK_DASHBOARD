export type LinearModel = {
  version: string
  features: string[]
  intercept: number
  coef: number[]
  norm?: {
    mean?: Record<string, number>
    std?: Record<string, number>
  }
}

function env(name: string, required = true): string | undefined {
  const v = (process.env as any)[name] as string | undefined
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

export async function getStorageObject(path: string) {
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string | undefined
  if (!SERVICE) throw new Error('Missing Supabase service key')
  const url = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${path.replace(/^\/+/, '')}`
  const r = await fetch(url, { headers: { Authorization: `Bearer ${SERVICE}`, apikey: SERVICE }, cache: 'no-store' })
  return r
}

// simple in-memory cache per model key
const MODEL_CACHE: Record<string, { loadedAt: number; model: LinearModel } | undefined> = {}
const MODEL_TTL_MS = 5 * 60 * 1000

export async function loadLinearModelFromStorage(path: string): Promise<LinearModel> {
  const now = Date.now()
  const cached = MODEL_CACHE[path]
  if (cached && now - cached.loadedAt < MODEL_TTL_MS) return cached.model

  // allow "latest.json" indirection file containing { path: ".../model_vX.json" }
  const r = await getStorageObject(path)
  if (!r.ok) throw new Error(`model fetch failed ${r.status}`)
  const txt = await r.text()
  let j: any
  try { j = JSON.parse(txt) } catch { throw new Error('model JSON parse error') }

  // If latest file points to actual model path
  if (j && typeof j === 'object' && j.path && typeof j.path === 'string' && (!j.features || !j.coef)) {
    const r2 = await getStorageObject(j.path)
    if (!r2.ok) throw new Error(`model indirection fetch failed ${r2.status}`)
    const txt2 = await r2.text()
    try { j = JSON.parse(txt2) } catch { throw new Error('model JSON parse error (indirect)') }
  }

  if (!Array.isArray(j?.features) || !Array.isArray(j?.coef) || typeof j?.intercept !== 'number') {
    throw new Error('invalid model schema')
  }
  const model: LinearModel = {
    version: String(j.version || 'v1'),
    features: j.features.map((x: any) => String(x)),
    intercept: Number(j.intercept),
    coef: j.coef.map((c: any) => Number(c)),
    norm: j.norm && typeof j.norm === 'object' ? {
      mean: j.norm.mean || {},
      std: j.norm.std || {},
    } : undefined,
  }
  if (model.features.length !== model.coef.length) throw new Error('model features/coef length mismatch')
  MODEL_CACHE[path] = { loadedAt: now, model }
  return model
}

export function sigmoid(z: number): number {
  if (!Number.isFinite(z)) return 0.5
  if (z > 20) return 1 - 1e-9
  if (z < -20) return 1e-9
  return 1 / (1 + Math.exp(-z))
}

