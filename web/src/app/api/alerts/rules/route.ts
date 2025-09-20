export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function json(obj: any, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8' },
  })
}

const RULES_KEY = 'alerts/rules.json'

type Rule = {
  id: string
  ticker: string
  type: 'BUY_ZONE' | 'STOP_TOUCH' | 'TARGET_TOUCH'
  enabled: boolean
}

async function readRules() {
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
  const url = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${RULES_KEY}`
  const r = await fetch(url, { headers: { Authorization: `Bearer ${SERVICE}`, apikey: SERVICE }, cache: 'no-store' })
  if (!r.ok) return [] as Rule[]
  try { return await r.json() as Rule[] } catch { return [] as Rule[] }
}

async function writeRules(rules: Rule[]) {
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
  const url = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${RULES_KEY}`
  const r = await fetch(url, {
    method: 'POST',
    headers: { Authorization: `Bearer ${SERVICE}`, apikey: SERVICE, 'x-upsert': 'true', 'content-type': 'application/json' },
    body: JSON.stringify(rules, null, 2),
  })
  return r.ok
}

function uuid() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return 'id-' + Math.random().toString(36).slice(2)
}

export async function GET() {
  try {
    const rules = await readRules()
    return json({ rules })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}))
    const incoming: Rule[] = Array.isArray(body?.rules) ? body.rules : []
    const cleaned: Rule[] = incoming.map((r) => ({
      id: r.id || uuid(),
      ticker: String(r.ticker || '').trim().toUpperCase(),
      type: (r.type as any) === 'STOP_TOUCH' ? 'STOP_TOUCH' : ((r.type as any) === 'TARGET_TOUCH' ? 'TARGET_TOUCH' : 'BUY_ZONE'),
      enabled: !!r.enabled,
    }))
    const ok = await writeRules(cleaned)
    return json({ ok, count: cleaned.length })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

