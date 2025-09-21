import { loadLinearModelFromStorage, sigmoid } from '../../../../lib/ml/model'
import { toBaseFeatures, vectorForModel, dot, featureContributions } from '../../../../lib/ml/features'

export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function json(obj: any, init?: number) { return new Response(JSON.stringify(obj), { status: init || 200, headers: { 'content-type': 'application/json' } }) }

function normalizeTicker(t: string) { return t.trim().toUpperCase().replace(/\s+/g, '_') }

async function fetchScreener(req: Request, tickers: string[]) {
  const u = new URL('/api/screener', req.url)
  u.searchParams.set('tickers', tickers.join(','))
  // force fresh data when called directly
  u.searchParams.set('refresh', '1')
  const r = await fetch(u.toString(), { cache: 'no-store' })
  if (!r.ok) throw new Error(`screener ${r.status}`)
  const j = await r.json()
  const items = Array.isArray(j?.items) ? j.items : []
  return items
}

// default model pointer in Storage
const DEFAULT_MODEL_POINTER = 'ml/models/latest.json'

async function findLatestModel(): Promise<string> {
  // Try to find the latest model by listing files and picking the newest timestamp
  try {
    const SUPABASE_URL = process.env.SUPABASE_URL!
    const SUPABASE_BUCKET = process.env.SUPABASE_BUCKET!
    const SERVICE = (process.env.SUPABASE_SERVICE_ROLE || process.env.SUPABASE_KEY) as string

    // List all model files in the ml/models directory
    const listUrl = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/list/${SUPABASE_BUCKET}`
    const response = await fetch(listUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${SERVICE}`,
        'apikey': SERVICE,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prefix: 'ml/models/',
        limit: 100
      })
    })

    if (response.ok) {
      const files = await response.json()
      // Find model files (not latest.json) and sort by timestamp in filename
      const modelFiles = files
        .filter((f: any) => f.name && f.name.startsWith('model_v') && f.name.endsWith('.json'))
        .sort((a: any, b: any) => b.name.localeCompare(a.name)) // Sort descending (newest first)

      if (modelFiles.length > 0) {
        return `ml/models/${modelFiles[0].name}`
      }
    }
  } catch (e) {
    console.warn('Failed to find latest model automatically:', e)
  }

  // Fallback to latest.json pointer
  return DEFAULT_MODEL_POINTER
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const tickers = (url.searchParams.get('tickers') || '').split(',').map(s => s.trim()).filter(Boolean).slice(0, 10)
    if (!tickers.length) return json({ error: 'tickers required' }, 400)

    const requestedModelPath = url.searchParams.get('model')
    const debug = url.searchParams.get('debug') === '1'

    // 1) Load model (cached in-memory across warm invocations)
    let model
    let modelError = null
    let modelPath = requestedModelPath || DEFAULT_MODEL_POINTER

    try {
      model = await loadLinearModelFromStorage(modelPath)
    } catch (e: any) {
      // If default latest.json fails, try to find the latest model automatically
      if (!requestedModelPath) {
        try {
          const autoModelPath = await findLatestModel()
          if (autoModelPath !== modelPath) {
            console.log(`Trying auto-discovered model: ${autoModelPath}`)
            modelPath = autoModelPath
            model = await loadLinearModelFromStorage(modelPath)
          }
        } catch (e2: any) {
          modelError = `${e?.message || String(e)} | Auto-discovery failed: ${e2?.message || String(e2)}`
        }
      }

      if (!model) {
        // Log the actual error for debugging
        modelError = modelError || e?.message || String(e)
        console.error('ML model load failed:', modelError)
        // Fallback: no model available -> degrade to heuristic based on screener score
        model = null
      }
    }

    // 2) Fetch features from screener
    const items = await fetchScreener(req, tickers)

    // 3) Predict
    const preds = items.map((x: any) => {
      const t = normalizeTicker(x?.ticker || '')
      if (!t) return { ticker: x?.ticker || '', p: null, note: 'missing_ticker' }
      if (!model) {
        const p = typeof x?.score === 'number' ? Math.max(0, Math.min(1, x.score / 100)) : null
        return { ticker: t, p, model: 'heuristic_score/100' }
      }
      const base = toBaseFeatures(x)
      const vec = vectorForModel(model, base)
      const z = dot(model, vec)
      const p = sigmoid(z)
      const contribs = featureContributions(model, base)
        .sort((a,b)=>Math.abs(b.contrib)-Math.abs(a.contrib))
        .slice(0, 5)
        .map(c=>({ key: c.key, contrib: c.contrib, weight: c.weight, mean: c.mean, std: c.std }))
      return { ticker: t, p, contribs }
    })

    const response = {
      generated_at: new Date().toISOString(),
      model: model ? { version: model.version, path: modelPath } : { version: 'heuristic' },
      count: preds.length,
      preds,
      ...(debug && modelError ? { debug: { modelError, modelPath } } : {})
    }
    return json(response)
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

