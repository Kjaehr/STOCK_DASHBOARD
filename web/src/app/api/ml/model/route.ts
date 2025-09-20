import { loadLinearModelFromStorage } from '../../../../lib/ml/model'

export const runtime = 'nodejs'

function json(obj: any, init?: number) { return new Response(JSON.stringify(obj), { status: init || 200, headers: { 'content-type': 'application/json' } }) }

// default model pointer in Storage
const DEFAULT_MODEL_POINTER = 'ml/models/latest.json'

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const modelPath = (url.searchParams.get('model') || DEFAULT_MODEL_POINTER).trim()
    const model = await loadLinearModelFromStorage(modelPath)
    const resp = {
      path: modelPath,
      version: model.version,
      features: model.features,
      intercept: model.intercept,
      coef_count: Array.isArray(model.coef) ? model.coef.length : 0,
      norm: {
        mean_keys: Object.keys(model.norm?.mean || {}),
        std_keys: Object.keys(model.norm?.std || {}),
      },
    }
    return json(resp)
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

