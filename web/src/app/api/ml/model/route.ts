import { loadLinearModelFromStorage } from '../../../../lib/ml/model'

export const runtime = 'nodejs'

function json(obj: any, init?: number) { return new Response(JSON.stringify(obj), { status: init || 200, headers: { 'content-type': 'application/json' } }) }

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
    const requestedModelPath = url.searchParams.get('model')
    let modelPath = requestedModelPath || DEFAULT_MODEL_POINTER

    let model
    try {
      model = await loadLinearModelFromStorage(modelPath)
    } catch (e: any) {
      // If default latest.json fails, try to find the latest model automatically
      if (!requestedModelPath) {
        const autoModelPath = await findLatestModel()
        if (autoModelPath !== modelPath) {
          modelPath = autoModelPath
          model = await loadLinearModelFromStorage(modelPath)
        }
      } else {
        throw e
      }
    }
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

