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

const LATEST_KEY = 'alerts/events/latest.json'

export async function GET() {
  try {
    const SUPABASE_URL = env('SUPABASE_URL')!
    const SUPABASE_BUCKET = env('SUPABASE_BUCKET')!
    const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
    const url = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${LATEST_KEY}`
    const r = await fetch(url, { headers: { Authorization: `Bearer ${SERVICE}`, apikey: SERVICE }, cache: 'no-store' })
    if (!r.ok) return json({ events: [] })
    const events = await r.json().catch(() => [])
    return json({ events })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

