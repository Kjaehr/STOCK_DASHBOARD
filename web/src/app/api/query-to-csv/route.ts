export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function bad(msg: string, code = 400) { return new Response(msg, { status: code, headers: { 'content-type': 'text/plain' } }) }

export async function GET(req: Request) {
  try {
    const SUPABASE_URL = env('SUPABASE_URL')!
    const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
    const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`

    const url = new URL(req.url)
    const table = (url.searchParams.get('table') || 'snapshots').replace(/[^a-zA-Z0-9_]/g, '')
    const qs = new URLSearchParams(url.searchParams)
    qs.delete('table')

    // Default selection
    if (!qs.has('select')) qs.set('select', '*')

    const upstream = await fetch(`${base}/${encodeURIComponent(table)}?${qs.toString()}`, {
      headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}`, Accept: 'text/csv' },
      cache: 'no-store',
    })

    if (!upstream.ok) {
      const txt = await upstream.text().catch(()=>'')
      return bad(`Upstream error ${upstream.status}: ${txt || 'failed'}`, 502)
    }

    // Stream CSV through
    const headers = new Headers()
    headers.set('content-type', 'text/csv; charset=utf-8')
    headers.set('cache-control', 'no-store')
    return new Response(upstream.body, { status: 200, headers })
  } catch (e: any) {
    return bad(String(e || 'error'), 500)
  }
}

