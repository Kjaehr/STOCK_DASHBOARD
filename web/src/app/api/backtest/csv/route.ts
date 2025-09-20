export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function bad(msg: string, code = 400) { return new Response(msg, { status: code, headers: { 'content-type': 'text/plain', 'cache-control': 'no-store' } }) }

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const id = (url.searchParams.get('id') || '').trim()
    if (!id) return bad('id required', 400)

    const SUPABASE_URL = env('SUPABASE_URL')!
    const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
    const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`

    const qs = new URLSearchParams()
    qs.set('run_id', `eq.${id}`)
    qs.set('select', 'ticker,entry_date,exit_date,entry_price,exit_price,r,pnl_pct,bars_held,exit_reason')

    const upstream = await fetch(`${base}/backtest_trades?${qs.toString()}`, {
      headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}`, Accept: 'text/csv' },
      cache: 'no-store',
    })
    if (!upstream.ok) return bad(`Upstream error ${upstream.status}: ${await upstream.text().catch(()=> '')}`, 502)

    const headers = new Headers()
    headers.set('content-type', 'text/csv; charset=utf-8')
    headers.set('cache-control', 'no-store')
    headers.set('content-disposition', `attachment; filename="backtest-${id}.csv"`)
    return new Response(upstream.body, { status: 200, headers })
  } catch (e: any) {
    return bad(String(e || 'error'), 500)
  }
}

