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

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const tickers = (url.searchParams.get('tickers') || '').split(',').map(s => s.trim()).filter(Boolean).slice(0, 10)
    if (!tickers.length) return json({ error: 'tickers required' }, 400)

    const modelPath = (url.searchParams.get('model') || DEFAULT_MODEL_POINTER).trim()

    // 1) Load model (cached in-memory across warm invocations)
    let model
    try {
      model = await loadLinearModelFromStorage(modelPath)
    } catch (e: any) {
      // Fallback: no model available -> degrade to heuristic based on screener score
      model = null
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

    return json({ generated_at: new Date().toISOString(), model: model ? { version: model.version, path: modelPath } : { version: 'heuristic' }, count: preds.length, preds })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

