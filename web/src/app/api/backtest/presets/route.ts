export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function json(obj: any, init?: number) { return new Response(JSON.stringify(obj), { status: init || 200, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } }) }

export async function GET() {
  try {
    const SUPABASE_URL = env('SUPABASE_URL')!
    const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
    const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`
    const qs = new URLSearchParams()
    qs.set('select','id,name,params,created_at')
    qs.set('order','created_at.desc')
    const r = await fetch(`${base}/backtest_presets?${qs.toString()}`, { headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}` }, cache: 'no-store' })
    if (!r.ok) return json({ error: 'fetch presets failed', status: r.status, text: await r.text().catch(()=>undefined) }, 502)
    const rows = await r.json().catch(()=>[])
    return json({ items: rows })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

export async function POST(req: Request){
  try {
    const body = await req.json().catch(()=> ({})) as any
    const name = (body?.name || '').trim()
    const params = body?.params
    if (!name || !params) return json({ error: 'name and params required' }, 400)
    const SUPABASE_URL = env('SUPABASE_URL')!
    const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
    const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`
    const r = await fetch(`${base}/backtest_presets`, {
      method: 'POST', headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}`, 'Prefer': 'return=representation', 'Content-Type':'application/json' },
      body: JSON.stringify([{ name, params }])
    })
    if (!r.ok) return json({ error: 'save preset failed', status: r.status, text: await r.text().catch(()=>undefined) }, 502)
    const rows = await r.json().catch(()=>[])
    return json({ ok: true, preset: rows?.[0] })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

