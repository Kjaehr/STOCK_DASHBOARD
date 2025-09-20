import type { NextRequest } from 'next/server'

export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function json(obj: any, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json; charset=utf-8' } })
}

function todayUTC() {
  const d = new Date()
  return d.toISOString().slice(0, 10)
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

type Rule = { id: string; ticker: string; type: 'BUY_ZONE' | 'STOP_TOUCH' | 'TARGET_TOUCH'; enabled: boolean }

type Event = { ts: string; ticker: string; type: string; title: string; details?: any; delivered?: boolean; channel?: string }

async function storageGet(path: string) {
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
  const url = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${path}`
  return await fetch(url, { headers: { Authorization: `Bearer ${SERVICE}`, apikey: SERVICE }, cache: 'no-store' })
}

async function storagePut(path: string, body: string | ArrayBuffer, contentType: string) {
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
  const url = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${path}`
  return await fetch(url, { method: 'POST', headers: { Authorization: `Bearer ${SERVICE}`, apikey: SERVICE, 'x-upsert': 'true', 'content-type': contentType }, body: body as any })
}

async function fetchRules(origin: string): Promise<Rule[]> {
  const r = await fetch(`${origin}/api/alerts/rules`, { cache: 'no-store' })
  const j = await r.json().catch(() => ({}))
  return Array.isArray(j?.rules) ? j.rules as Rule[] : []
}

function buildEvent(ticker: string, type: Rule['type'], item: any): Event | null {
  const price = Number(item?.price)
  const stop = Number(item?.exit_levels?.stop_suggest)
  const t1 = Array.isArray(item?.exit_levels?.targets) ? Number(item.exit_levels.targets[0]) : NaN
  if (type === 'BUY_ZONE') {
    if (item?.position_health?.in_buy_zone) {
      return { ts: new Date().toISOString(), ticker, type, title: `${ticker}: Entered buy zone`, details: { price, in_zone: true } }
    }
  } else if (type === 'STOP_TOUCH') {
    if (Number.isFinite(price) && Number.isFinite(stop) && price <= stop) {
      return { ts: new Date().toISOString(), ticker, type, title: `${ticker}: Stop touched`, details: { price, stop } }
    }
  } else if (type === 'TARGET_TOUCH') {
    if (Number.isFinite(price) && Number.isFinite(t1) && price >= t1) {
      return { ts: new Date().toISOString(), ticker, type, title: `${ticker}: Target1 touched`, details: { price, target: t1 } }
    }
  }
  return null
}

async function deliverEmailIfConfigured(ev: Event) {
  const apiKey = process.env.RESEND_API_KEY || ''
  const to = process.env.ALERT_EMAIL_TO || ''
  const from = process.env.ALERT_EMAIL_FROM || 'alerts@stockdash.local'
  if (!apiKey || !to) return { delivered: false, reason: 'email not configured' }
  const r = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'content-type': 'application/json' },
    body: JSON.stringify({ from, to, subject: `[Alert] ${ev.title}`, text: JSON.stringify(ev, null, 2) }),
  })
  const ok = r.status < 300
  return { delivered: ok, status: r.status, text: ok ? undefined : await r.text().catch(() => undefined) }
}

export async function GET(req: NextRequest) {
  const url = new URL(req.url)
  const origin = `${url.protocol}//${url.host}`
  const dataBase = process.env.NEXT_PUBLIC_DATA_BASE || '/api/data'
  const batchSize = Math.max(1, parseIntSafe(process.env.ALERTS_BATCH_SIZE || null, 10) || 10)

  try {
    // Load tickers
    const metaRes = await fetch(`${origin}${dataBase.replace(/\/$/, '')}/meta.json`, { cache: 'no-store' })
    if (!metaRes.ok) return json({ ok: false, error: 'meta fetch failed', status: metaRes.status }, 502)
    const meta = await metaRes.json().catch(() => ({})) as any
    const tickers: string[] = Array.isArray(meta?.tickers) ? meta.tickers : []
    if (!tickers.length) return json({ ok: true, note: 'no tickers' })

    const rules = (await fetchRules(origin)).filter(r => r.enabled)
    if (!rules.length) return json({ ok: true, note: 'no rules' })

    // Index rules per ticker
    const map = new Map<string, Rule[]>()
    for (const r of rules) {
      const t = r.ticker.trim().toUpperCase()
      if (!map.has(t)) map.set(t, [])
      map.get(t)!.push(r)
    }

    const targets = tickers.filter(t => map.has(t))
    const chunks = chunk(targets, Math.min(batchSize, 10))

    const events: Event[] = []
    for (const ch of chunks) {
      const r = await fetch(`${origin}/api/screener?` + new URLSearchParams({ tickers: ch.join(',') }).toString())
      const j = await r.json().catch(() => ({})) as any
      const items: any[] = Array.isArray(j?.items) ? j.items : []
      for (const it of items) {
        const t = String(it?.ticker || '').toUpperCase()
        const rs = map.get(t) || []
        for (const rule of rs) {
          const ev = buildEvent(t, rule.type, it)
          if (ev) events.push(ev)
        }
      }
    }

    // de-dupe per 24h using Storage flags
    const day = todayUTC()
    const delivered: Event[] = []
    for (const ev of events) {
      const dedupeKey = `alerts/dedupe/${day}/${ev.ticker}-${ev.type}.flag`
      const head = await storageGet(dedupeKey)
      if (head.ok) continue // already delivered today
      // Deliver (email if configured; else log only)
      const res = await deliverEmailIfConfigured(ev).catch(() => ({ delivered: false }))
      ev.delivered = !!res.delivered
      ev.channel = res.delivered ? 'email' : 'log'
      await storagePut(dedupeKey, JSON.stringify({ ts: ev.ts, ev }), 'application/json')
      // Store event for history
      const evKey = `alerts/events/${day}/${ev.ts.replace(/[:.]/g,'-')}_${ev.ticker}_${ev.type}.json`
      await storagePut(evKey, JSON.stringify(ev, null, 2), 'application/json')
      delivered.push(ev)
    }

    // Update latest manifest (best-effort)
    const latestKey = 'alerts/events/latest.json'
    const latest = await storageGet(latestKey).then(r => r.ok ? r.json() : [] as any[]).catch(() => [] as any[])
    const merged = [...delivered, ...(Array.isArray(latest) ? latest : [])].slice(0, 100)
    await storagePut(latestKey, JSON.stringify(merged, null, 2), 'application/json')

    return json({ ok: true, checked: targets.length, candidates: events.length, delivered: delivered.length })
  } catch (e: any) {
    return json({ ok: false, error: e?.message || String(e) }, 500)
  }
}

