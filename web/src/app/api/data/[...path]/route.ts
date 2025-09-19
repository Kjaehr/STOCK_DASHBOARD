export const runtime = 'edge'

function env(name: string): string {
  const v = process.env[name]
  if (!v) throw new Error(`Missing env: ${name}`)
  return v
}

const TTL_SECONDS = Number(process.env.DATA_TTL_SECONDS || '900')

function bypass(req: Request): boolean {
  try {
    const u = new URL(req.url)
    return u.searchParams.get('refresh') === '1' || u.searchParams.has('t')
  } catch {
    return false
  }
}

// Note: Avoid strict typing of the route handler context to satisfy Next.js build validation across versions
export async function GET(req: Request, context: any) {
  const SUPABASE_URL = env('SUPABASE_URL')
  const SUPABASE_KEY = env('SUPABASE_KEY') // Use anon or service; do NOT expose on client
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET')

  const segments = (context?.params?.path ?? []) as string[]
  if (!segments.length) {
    return new Response(JSON.stringify({ error: 'Missing object path (e.g. /api/data/meta.json)' }), { status: 400, headers: { 'content-type': 'application/json' } })
  }
  // Normalize to prevent traversal
  const key = segments.map(s => s.replace(/\\/g, '/').replace(/^\/+|\/+$/g, '')).filter(Boolean).join('/')
  // Storage REST (works for public or private when using Authorization)
  const storageUrl = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${key}`

  const noStore = bypass(req)

  const upstream = await fetch(storageUrl, noStore ? {
    headers: {
      Authorization: `Bearer ${SUPABASE_KEY}`,
    },
    cache: 'no-store',
  } : {
    headers: {
      Authorization: `Bearer ${SUPABASE_KEY}`,
    },
    // Enable Next.js caching on the edge
    next: { revalidate: TTL_SECONDS },
  })

  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '')
    const headers: Record<string, string> = {
      'content-type': upstream.headers.get('content-type') || 'text/plain',
    }
    if (noStore) {
      headers['cache-control'] = 'no-store'
    } else {
      headers['cache-control'] = `s-maxage=${Math.max(5, Math.floor(TTL_SECONDS/3))}, stale-while-revalidate=${TTL_SECONDS}`
    }
    return new Response(text || `Upstream error ${upstream.status}`, {
      status: upstream.status,
      headers,
    })
  }

  const hdrs = new Headers()
  // Forward useful headers
  const ct = upstream.headers.get('content-type') || 'application/json; charset=utf-8'
  hdrs.set('content-type', ct)
  const etag = upstream.headers.get('etag')
  if (etag) hdrs.set('etag', etag)
  // Cache behavior
  if (noStore) {
    hdrs.set('cache-control', 'no-store')
  } else {
    // Cache at the edge; allow clients to cache briefly too
    hdrs.set('cache-control', `public, max-age=30, s-maxage=${TTL_SECONDS}, stale-while-revalidate=${TTL_SECONDS * 5}`)
  }

  return new Response(upstream.body, {
    status: 200,
    headers: hdrs,
  })
}

