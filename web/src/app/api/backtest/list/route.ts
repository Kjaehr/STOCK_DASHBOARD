export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function json(obj: any, init?: number, extraHeaders?: Record<string,string>) {
  return new Response(JSON.stringify(obj), { status: init || 200, headers: { 'content-type': 'application/json', 'cache-control': 'no-store', ...(extraHeaders||{}) } })
}

export async function GET(req: Request) {
  try {
    const SUPABASE_URL = env('SUPABASE_URL')!
    const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
    const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`
    const url = new URL(req.url)

    const pageSize = Math.max(1, Math.min(100, Number(url.searchParams.get('pageSize') ?? 10)))
    const page = Math.max(1, Number(url.searchParams.get('page') ?? 1))
    const offset = (page - 1) * pageSize
    const ticker = (url.searchParams.get('ticker') || '').trim().toUpperCase()
    const from = (url.searchParams.get('from') || '').trim()
    const to = (url.searchParams.get('to') || '').trim()

    const qs = new URLSearchParams()
    qs.set('select', 'id,created_at,tickers,lookback_days,summary')
    qs.set('order', 'created_at.desc')
    qs.set('limit', String(pageSize))
    qs.set('offset', String(offset))
    if (ticker) qs.set('tickers', `ov.{${ticker}}`)
    if (from) qs.set('created_at', `gte.${from}`)
    if (to) qs.append('created_at', `lte.${to}`)

    const headers: Record<string,string> = { apikey: SERVICE, Authorization: `Bearer ${SERVICE}`, Prefer: 'count=exact' }
    const rs = await fetch(`${base}/backtest_runs?${qs.toString()}`, { headers, cache: 'no-store' })
    if (!rs.ok) return json({ error: 'fetch list failed', status: rs.status, text: await rs.text().catch(()=>undefined) }, 502)
    const rows = await rs.json().catch(()=>[])
    const cr = rs.headers.get('content-range') || '' // e.g. 0-9/42
    const total = Number(cr.split('/')[1] || rows.length)
    return json({ items: rows, total, page, pageSize })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

