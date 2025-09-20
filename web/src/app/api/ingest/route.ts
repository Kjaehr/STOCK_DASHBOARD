import crypto from 'crypto'
import { ParquetSchema, ParquetWriter } from 'parquetjs-lite'
import { Writable } from 'stream'

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

async function putStorageBinary(path: string, data: ArrayBuffer, upsert = true, contentType = 'application/octet-stream') {
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

    // 4) Parquet per run (partitioned path)
    const parquetEnabled = (process.env.PARQUET_ENABLED ?? '1') === '1'
    if (parquetEnabled) {
      const pqBase = 'stocks-datalake/parquet'
      const d = new Date(nowIso)
      const yyyy = d.getUTCFullYear()
      const mm = String(d.getUTCMonth() + 1).padStart(2, '0')
      const dd = String(d.getUTCDate()).padStart(2, '0')

      const schema = new ParquetSchema({
        timestamp_iso: { type: 'UTF8' },
        ticker: { type: 'UTF8' },
        price: { type: 'DOUBLE', optional: true },
        score: { type: 'INT64', optional: true },
        rsi: { type: 'DOUBLE', optional: true },
        atr_pct: { type: 'DOUBLE', optional: true },
        sma20: { type: 'DOUBLE', optional: true },
        sma50: { type: 'DOUBLE', optional: true },
        sma200: { type: 'DOUBLE', optional: true },
        sector: { type: 'UTF8', optional: true },
        industry: { type: 'UTF8', optional: true },
        sent_mean7: { type: 'DOUBLE', optional: true },
        sent_count7: { type: 'INT64', optional: true },
        flags: { type: 'UTF8', optional: true },
        in_buy_zone: { type: 'BOOLEAN', optional: true },
        dist_to_stop_pct: { type: 'DOUBLE', optional: true },
        dist_to_t1_pct: { type: 'DOUBLE', optional: true },
      } as any)

      for (const x of items) {
        const t = normalizeTicker(x.ticker)
        const pqKey = `${pqBase}/ticker=${t}/year=${yyyy}/month=${mm}/day=${dd}/snapshot.parquet`

        class CollectWritable extends Writable {
          chunks: Buffer[] = []
          _write(chunk: any, _enc: any, cb: any) {
            this.chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
            cb()
          }
        }
        const sink = new CollectWritable()
        const writer = await ParquetWriter.openStream(schema as any, sink as any)
        await writer.appendRow({
          timestamp_iso: nowIso,
          ticker: x.ticker,
          price: x.price ?? null,
          score: x.score ?? null,
          rsi: x?.technicals?.rsi ?? null,
          atr_pct: x?.technicals?.atr_pct ?? null,
          sma20: x?.technicals?.sma20 ?? null,
          sma50: x?.technicals?.sma50 ?? null,
          sma200: x?.technicals?.sma200 ?? null,
          sector: x?.fundamentals?.sector ?? null,
          industry: x?.fundamentals?.industry ?? null,
          sent_mean7: x?.sentiment?.mean7 ?? null,
          sent_count7: x?.sentiment?.count7 ?? null,
          flags: Array.isArray(x?.flags) ? x.flags.join('|') : null,
          in_buy_zone: x?.position_health?.in_buy_zone ?? null,
          dist_to_stop_pct: x?.position_health?.dist_to_stop_pct ?? null,
          dist_to_t1_pct: x?.position_health?.dist_to_t1_pct ?? null,
        })
        await writer.close()

        const buf = Buffer.concat(sink.chunks)
        const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
        const put = await putStorageBinary(pqKey, ab, true, 'application/octet-stream')
        if (!put.ok) {
          const txt = await put.text().catch(()=>'')
          console.error('Parquet upload failed', pqKey, put.status, txt)
        }
        manifestFiles.push({ path: pqKey, size: buf.length, checksum: sha256Hex(buf) })
      }
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

