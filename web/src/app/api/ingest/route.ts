import crypto from 'crypto'

export const runtime = 'nodejs'

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function json(obj: any, init?: number) { return new Response(JSON.stringify(obj), { status: init || 200, headers: { 'content-type': 'application/json' } }) }

function normalizeTicker(t: string) { return t.trim().toUpperCase().replace(/\s+/g, '_') }

async function fetchScreener(req: Request, tickers: string[]) {
  const u = new URL('/api/screener', req.url)
  u.searchParams.set('tickers', tickers.join(','))
  u.searchParams.set('refresh', '1')
  const r = await fetch(u.toString(), { cache: 'no-store' })
  if (!r.ok) throw new Error(`screener ${r.status}`)
  const j = await r.json()
  const items = Array.isArray(j?.items) ? j.items : []
  return items
}

function buildCsvHeader() {
  return [
    'timestamp_iso','ticker','price','score','rsi','atr_pct','sma20','sma50','sma200',
    'sector','industry','sent_mean7','sent_count7','flags','in_buy_zone','dist_to_stop_pct','dist_to_t1_pct'
  ].join(',') + '\n'
}

function toCsvRow(ts: string, x: any) {
  const get = (v: any) => (v == null ? '' : String(v))
  const flags = Array.isArray(x?.flags) ? x.flags.join('|') : ''
  const t = [
    ts,
    x?.ticker,
    x?.price,
    x?.score,
    x?.technicals?.rsi,
    x?.technicals?.atr_pct,
    x?.technicals?.sma20,
    x?.technicals?.sma50,
    x?.technicals?.sma200,
    x?.fundamentals?.sector,
    x?.fundamentals?.industry,
    x?.sentiment?.mean7,
    x?.sentiment?.count7,
    flags,
    x?.position_health?.in_buy_zone,
    x?.position_health?.dist_to_stop_pct,
    x?.position_health?.dist_to_t1_pct,
  ].map(get)
  return t.join(',') + '\n'
}

async function upsertSnapshotsToPostgres(rows: any[]) {
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
  if (!SUPABASE_URL || !SERVICE) return { ok: false, reason: 'missing supabase env' }
  const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`
  const res = await fetch(`${base}/snapshots?on_conflict=ticker,timestamp_iso`, {
    method: 'POST',
    headers: {
      apikey: SERVICE,
      Authorization: `Bearer ${SERVICE}`,
      'Prefer': 'resolution=merge-duplicates',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(rows),
  })
  const ok = res.status < 300
  return { ok, status: res.status, text: ok ? undefined : await res.text().catch(()=>undefined) }
}

async function putStorageObject(path: string, data: string, upsert = true, contentType: string = 'text/plain; charset=utf-8') {
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
  const url = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${path.replace(/^\/+/, '')}`
  const r = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${SERVICE}`,
      apikey: SERVICE,
      ...(upsert ? { 'x-upsert': 'true' } : {}),
      'Content-Type': contentType,
    },
    body: data,
  })
  return r
}

async function getStorageObject(path: string) {
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
  const url = `${SUPABASE_URL.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET)}/${path.replace(/^\/+/, '')}`
  const r = await fetch(url, { headers: { Authorization: `Bearer ${SERVICE}`, apikey: SERVICE }, cache: 'no-store' })
  return r
}

function sha256Hex(buf: Uint8Array | string) {
  const h = crypto.createHash('sha256')
  if (typeof buf === 'string') h.update(buf)
  else h.update(buf)
  return h.digest('hex')
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const tickers = (url.searchParams.get('tickers') || '').split(',').map(s => s.trim()).filter(Boolean).slice(0, 10)
    if (!tickers.length) return json({ error: 'tickers required' }, 400)

    // 1) Fetch current snapshots from screener (live)
    const items = await fetchScreener(req, tickers)
    const nowIso = new Date().toISOString()

    // 2) Upsert to Postgres (best-effort)
    const pgRows = items.map((x: any) => ({
      timestamp_iso: nowIso,
      ticker: x.ticker,
      price: x.price,
      fundamentals: x.fundamentals || null,
      technicals: x.technicals || null,
      sentiment: x.sentiment || null,
      score: x.score || null,
      flags: x.flags || [],
      position_health: x.position_health || null,
    }))
    const pgRes = await upsertSnapshotsToPostgres(pgRows).catch(() => ({ ok: false }))

    // 3) CSV append per ticker in Storage (stocks-datalake/csv/*.csv)
    const csvBase = 'stocks-datalake/csv'
    const manifestFiles: Array<{ path: string; size?: number; checksum?: string }> = []

    for (const x of items) {
      const t = normalizeTicker(x.ticker)
      const key = `${csvBase}/${t}.csv`
      // Read existing (if any)
      let existing = ''
      const head = await getStorageObject(key)
      if (head.ok) {
        existing = await head.text().catch(() => '')
      }
      let out = existing
      if (!existing) out += buildCsvHeader()
      const line = toCsvRow(nowIso, x)
      out += line
      const put = await putStorageObject(key, out, true, 'text/csv; charset=utf-8')
      if (!put.ok) {
        const txt = await put.text().catch(()=> '')
        console.error('CSV upload failed', key, put.status, txt)
      }
      const checksum = sha256Hex(line)
      manifestFiles.push({ path: key, size: out.length, checksum })
    }

    // 4) Parquet per run (partitioned path) — requires dependency; skip if not enabled
    const parquetEnabled = process.env.PARQUET_ENABLED === '1'
    if (!parquetEnabled) {
      // noop; we will still write manifest
    } else {
      // Placeholder: Implement when dependency approved
    }

    // 5) Write manifest latest.json
    const manifest = { generated_at: nowIso, files: manifestFiles }
    const manKey = 'stocks-datalake/manifests/latest.json'
    const manPut = await putStorageObject(manKey, JSON.stringify(manifest, null, 2), true, 'application/json; charset=utf-8')
    if (!manPut.ok) {
      console.error('Manifest upload failed', manPut.status, await manPut.text().catch(()=>''))
    }

    return json({ ok: true, count: items.length, postgres: pgRes?.ok ?? false, manifest_key: manKey })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

