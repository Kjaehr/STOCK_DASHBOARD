import type { NextRequest } from 'next/server'

export const runtime = 'nodejs'

function json(obj: any, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8' },
  })
}

function parseIntSafe(v: string | null, def: number) {
  const n = v != null ? parseInt(v, 10) : NaN
  return Number.isFinite(n) ? n : def
}

export async function GET(req: NextRequest) {
  const url = new URL(req.url)
  const origin = `${url.protocol}//${url.host}`
  const dataBase = process.env.NEXT_PUBLIC_DATA_BASE || '/api/data'
  const n = Math.max(1, parseIntSafe(url.searchParams.get('n'), parseInt(process.env.WARM_SCREENER_N || '10', 10) || 10))

  try {
    // 1) Load meta for tickers list
    const metaRes = await fetch(`${origin}${dataBase.replace(/\/$/, '')}/meta.json`, { cache: 'no-store' })
    if (!metaRes.ok) {
      console.error('warm: meta fetch failed', metaRes.status)
      return json({ ok: false, error: 'meta fetch failed', status: metaRes.status }, 502)
    }
    const meta = await metaRes.json().catch(() => ({})) as any
    const tickers: string[] = Array.isArray(meta?.tickers) ? meta.tickers : []
    if (!tickers.length) return json({ ok: false, error: 'no tickers' }, 400)

    const list = tickers.slice(0, n)
    const qs = new URLSearchParams({ tickers: list.join(',') })

    // 2) Call screener to set edge cache (no refresh)
    const r = await fetch(`${origin}/api/screener?${qs.toString()}`)
    const status = r.status
    let info: any = null
    try { info = await r.json() } catch {}
    if (status === 429 || status >= 500) {
      console.warn('warm: screener possibly rate-limited/error', { status, info })
    }

    return json({ ok: status < 300, warmed: list.length, status })
  } catch (e: any) {
    console.error('warm: fatal', e?.message || String(e))
    return json({ ok: false, error: e?.message || String(e) }, 500)
  }
}

