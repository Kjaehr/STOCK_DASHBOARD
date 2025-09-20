export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function json(obj: any, init?: number) { return new Response(JSON.stringify(obj), { status: init || 200, headers: { 'content-type': 'application/json' } }) }

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const ticker = (url.searchParams.get('ticker') || '').trim()
    const fmt = (url.searchParams.get('fmt') || 'csv').toLowerCase()
    if (!ticker) return json({ error: 'ticker required' }, 400)
    if (fmt !== 'csv') return json({ error: 'unsupported fmt' }, 400)

    const SUPABASE_URL = env('SUPABASE_URL')!
    const SUPABASE_BUCKET = env('SUPABASE_BUCKET')!
    const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string

    const key = `stocks-datalake/csv/${ticker.replace(/\s+/g,'_').toUpperCase()}.csv`
    const signUrl = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/sign/${encodeURIComponent(SUPABASE_BUCKET)}/${key}`

    const rs = await fetch(signUrl, {
      method: 'POST',
      headers: { 'content-type': 'application/json', apikey: SERVICE, Authorization: `Bearer ${SERVICE}` },
      body: JSON.stringify({ expiresIn: 3600, download: true, filename: `${ticker}.csv` }),
    })
    if (!rs.ok) return json({ error: 'sign failed', status: rs.status, text: await rs.text().catch(()=>undefined) }, 500)
    const j = await rs.json().catch(()=>null) as any
    const urlSigned = j?.signedUrl || j?.signedURL || j?.url
    if (!urlSigned) return json({ error: 'malformed sign response' }, 500)

    return json({ url: urlSigned, key })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

