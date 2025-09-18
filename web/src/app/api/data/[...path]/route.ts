export const runtime = 'edge'

function env(name: string): string {
  const v = process.env[name]
  if (!v) throw new Error(`Missing env: ${name}`)
  return v
}

const TTL_SECONDS = Number(process.env.DATA_TTL_SECONDS || '60')

export async function GET(req: Request, ctx: { params: { path?: string[] } }) {
  const SUPABASE_URL = env('SUPABASE_URL')
  const SUPABASE_KEY = env('SUPABASE_KEY') // Use anon or service; do NOT expose on client
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET')

  const segments = (ctx.params.path || [])
  if (!segments.length) {
    return new Response(JSON.stringify({ error: 'Missing object path (e.g. /api/data/meta.json)' }), { status: 400, headers: { 'content-type': 'application/json' } })
  }
  // Normalize to prevent traversal
  const key = segments.map(s => s.replace(/\\/g, '/').replace(/^\/+|\/+$/g, '')).filter(Boolean).join('/')
  // Storage REST (works for public or private when using Authorization)
  const storageUrl = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${key}`

  const upstream = await fetch(storageUrl, {
    headers: {
      Authorization: `Bearer ${SUPABASE_KEY}`,
    },
    // Enable Next.js caching on the edge
    next: { revalidate: TTL_SECONDS },
  })

  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '')
    return new Response(text || `Upstream error ${upstream.status}`, {
      status: upstream.status,
      headers: {
        'cache-control': `s-maxage=${Math.max(5, Math.floor(TTL_SECONDS/3))}, stale-while-revalidate=${TTL_SECONDS}`,
        'content-type': upstream.headers.get('content-type') || 'text/plain',
      },
    })
  }

  const hdrs = new Headers()
  // Forward useful headers
  const ct = upstream.headers.get('content-type') || 'application/json; charset=utf-8'
  hdrs.set('content-type', ct)
  const etag = upstream.headers.get('etag')
  if (etag) hdrs.set('etag', etag)
  // Cache at the edge; allow clients to cache briefly too
  hdrs.set('cache-control', `public, max-age=30, s-maxage=${TTL_SECONDS}, stale-while-revalidate=${TTL_SECONDS * 5}`)

  return new Response(upstream.body, {
    status: 200,
    headers: hdrs,
  })
}

