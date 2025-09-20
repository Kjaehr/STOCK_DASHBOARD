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

function chunk<T>(arr: T[], size: number) {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size))
  return out
}

function selectGroup<T>(arr: T[], group: number, groups: number) {
  const out: T[] = []
  for (let i = 0; i < arr.length; i++) if (i % groups === group) out.push(arr[i])
  return out
}

export async function GET(req: NextRequest) {
  const started = Date.now()
  const url = new URL(req.url)
  const group = parseIntSafe(url.searchParams.get('group'), 0)
  const groups = Math.max(1, parseIntSafe(url.searchParams.get('groups'), parseInt(process.env.INGEST_GROUPS || '3', 10) || 3))
  const maxPerIngest = Math.max(1, parseIntSafe(process.env.INGEST_MAX_PER_CALL || null, 10) || 10)
  const budgetMs = Math.max(10_000, parseIntSafe(process.env.CRON_TIME_BUDGET_MS || null, 50_000))

  // Resolve origin of this request to call sibling routes
  const origin = `${url.protocol}//${url.host}`
  const dataBase = process.env.NEXT_PUBLIC_DATA_BASE || '/api/data'

  try {
    // 1) Load current ticker universe from meta.json
    const metaRes = await fetch(`${origin}${dataBase.replace(/\/$/, '')}/meta.json`, { cache: 'no-store' })
    if (!metaRes.ok) {
      console.error('cron: meta fetch failed', metaRes.status)
      return json({ ok: false, error: 'meta fetch failed', status: metaRes.status }, 502)
    }
    const meta = await metaRes.json().catch(() => ({})) as any
    const tickers: string[] = Array.isArray(meta?.tickers) ? meta.tickers : []
    if (!tickers.length) return json({ ok: false, error: 'no tickers in meta' }, 400)

    // 2) Select this group's slice (round-robin)
    const mine = selectGroup(tickers, group, groups)
    if (!mine.length) return json({ ok: true, note: 'empty group (no assigned tickers)', group, groups })

    // 3) Call /api/ingest in chunks of maxPerIngest, respecting time budget
    const chunks = chunk(mine, Math.max(1, Math.min(maxPerIngest, 15)))
    const results: Array<{ batch: number; status: number; count: number }> = []

    for (let i = 0; i < chunks.length; i++) {
      const elapsed = Date.now() - started
      if (elapsed > budgetMs) {
        console.warn('cron: time budget exceeded, stopping early', { handled: results.length, total: chunks.length })
        break
      }
      const qs = new URLSearchParams({ tickers: chunks[i].join(',') })
      const r = await fetch(`${origin}/api/ingest?${qs.toString()}`, { cache: 'no-store' })
      const status = r.status
      const body: any = await r.json().catch(() => ({}))
      if (status === 429 || status >= 500) {
        console.warn('cron: ingest upstream possibly rate-limited/error', { status, body })
      }
      results.push({ batch: i, status, count: Array.isArray(chunks[i]) ? chunks[i].length : 0 })
    }

    return json({ ok: true, group, groups, assigned: mine.length, invoked: results.length, results })
  } catch (e: any) {
    console.error('cron: fatal error', e?.message || String(e))
    return json({ ok: false, error: e?.message || String(e) }, 500)
  }
}

