export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function json(obj: any, init?: number) { return new Response(JSON.stringify(obj), { status: init || 200, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } }) }

export async function GET(req: Request) {
  try {
    const SUPABASE_URL = env('SUPABASE_URL')!
    const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
    const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`
    const url = new URL(req.url)
    const limit = Math.max(1, Math.min(100, Number(url.searchParams.get('limit') ?? 20)))
    const qs = new URLSearchParams()
    qs.set('select', 'id,created_at,tickers,lookback_days,summary')
    qs.set('order', 'created_at.desc')
    qs.set('limit', String(limit))
    const rs = await fetch(`${base}/backtest_runs?${qs.toString()}`, { headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}` }, cache: 'no-store' })
    if (!rs.ok) return json({ error: 'fetch list failed', status: rs.status, text: await rs.text().catch(()=>undefined) }, 502)
    const rows = await rs.json().catch(()=>[])
    return json({ items: rows })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

