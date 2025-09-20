import yahooFinance from 'yahoo-finance2'
import { backtestOne, aggregateResults, type StrategyParams, type OHLC } from '../../../lib/backtest'

export const runtime = 'nodejs'

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const id = (url.searchParams.get('id') || '').trim()
    if (!id) return json({ error: 'id required' }, 400)
    const SUPABASE_URL = env('SUPABASE_URL')!
    const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
    const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`
    const runRes = await fetch(`${base}/backtest_runs?id=eq.${encodeURIComponent(id)}&select=*`, { headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}` }, cache: 'no-store' })
    if (!runRes.ok) return json({ error: 'fetch run failed', status: runRes.status, text: await runRes.text().catch(()=>undefined) }, 502)
    const runs = await runRes.json().catch(()=>[])
    const run = runs?.[0]
    if (!run) return json({ error: 'not found' }, 404)
    const trRes = await fetch(`${base}/backtest_trades?run_id=eq.${encodeURIComponent(id)}&select=*`, { headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}` }, cache: 'no-store' })
    const trades = trRes.ok ? await trRes.json().catch(()=>[]) : []
    return json({ run, trades })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

function json(obj: any, init?: number) { return new Response(JSON.stringify(obj), { status: init || 200, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } }) }

async function fetchHistory(ticker: string, lookbackDays: number): Promise<OHLC[]> {
  const period1 = new Date(Date.now() - lookbackDays * 24 * 60 * 60 * 1000)
  const ch: any = await (yahooFinance as any).chart(ticker, { period1, interval: '1d' } as any)
  let hist: any[] = Array.isArray(ch?.quotes) && ch.quotes.length ? ch.quotes : []
  if (!hist.length) {
    const ts: number[] = ch?.timestamp || []
    const q = ch?.indicators?.quote?.[0] || {}
    hist = ts.map((tsVal: number, i: number) => ({
      date: new Date(tsVal * 1000),
      close: q?.close?.[i],
      high: q?.high?.[i],
      low: q?.low?.[i],
    })).filter((x: any) => Number.isFinite(x.close) && Number.isFinite(x.high) && Number.isFinite(x.low))
  }
  return hist.map((x: any) => ({ date: (x.date ? new Date(x.date).toISOString().slice(0,10) : ''), close: Number(x.close), high: Number(x.high), low: Number(x.low) }))
}

async function insertRun(tickers: string[], lookbackDays: number, params: any, summary: any, equity: any) {
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
  const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`
  const res = await fetch(`${base}/backtest_runs`, {
    method: 'POST',
    headers: {
      apikey: SERVICE,
      Authorization: `Bearer ${SERVICE}`,
      'Prefer': 'return=representation',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify([{ tickers, lookback_days: lookbackDays, params, status: 'completed', summary, equity }]),
  })
  if (!res.ok) throw new Error(`insert run failed: ${res.status} ${await res.text().catch(()=> '')}`)
  const rows = await res.json().catch(()=> [])
  const id = rows?.[0]?.id
  if (!id) throw new Error('insert run: missing id in response')
  return id as string
}

async function insertTrades(run_id: string, trades: any[]) {
  if (!trades.length) return
  const SUPABASE_URL = env('SUPABASE_URL')!
  const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
  const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`
  const rows = trades.map(t => ({ run_id, ...t }))
  const res = await fetch(`${base}/backtest_trades`, {
    method: 'POST',
    headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(rows),
  })
  if (!res.ok) throw new Error(`insert trades failed: ${res.status} ${await res.text().catch(()=> '')}`)
}

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(()=> ({})) as any
    const rawTickers = Array.isArray(body?.tickers) ? body.tickers : String(body?.tickers || '').split(',')
    const tickers = rawTickers.map((t: any) => String(t || '').trim().toUpperCase()).filter(Boolean).slice(0, 20)
    if (!tickers.length) return json({ error: 'tickers required' }, 400)

    // lookbackYears 1..5 => days (add small buffer)
    const lookbackYears = Math.max(1, Math.min(5, Number(body?.lookbackYears ?? 2)))
    const lookbackDays = Math.round(lookbackYears * 365 + 7)

    const params: StrategyParams = body?.params || {}

    // Fetch data with small concurrency
    async function pLimit<T>(n: number, tasks: (()=>Promise<T>)[]) {
      const out: T[] = new Array(tasks.length)
      let i = 0
      async function run() { for(;;){ const idx = i++; if (idx >= tasks.length) break; out[idx] = await tasks[idx]() } }
      const workers = Array.from({length: Math.min(n, tasks.length)}, () => run())
      await Promise.all(workers)
      return out
    }

    const tasks = tickers.map((t: string) => async () => {
      const hist = await fetchHistory(t, lookbackDays)
      if (!hist?.length) return { ticker: t, error: 'no data', summary: null, equityCurve: [], trades: [] }
      return backtestOne(t, hist, params)
    })

    const results = await pLimit(4, tasks)
    const perTicker = results.filter((r: any) => !('error' in r))
    const agg = aggregateResults(perTicker as any)

    // Flatten trades for persistence
    const flatTrades = (perTicker as any[]).flatMap(r => r.trades.map((t: any) => ({
      ticker: r.ticker,
      entry_date: t.entry_date,
      exit_date: t.exit_date,
      entry_price: t.entry_price,
      exit_price: t.exit_price,
      r: t.r,
      pnl_pct: t.pnl_pct,
      bars_held: t.bars_held,
      exit_reason: t.exit_reason,
    })))

    let run_id: string | undefined
    try {
      run_id = await insertRun(tickers, lookbackDays, params, { ...agg.summary, perTicker: perTicker.map((r: any) => ({ ticker: r.ticker, ...r.summary })) }, agg.equityCurve)
      await insertTrades(run_id, flatTrades)
    } catch (e) {
      // If Supabase is not configured, we still return results
      console.warn('backtest: persist skipped/failed:', (e as any)?.message || String(e))
    }

    return json({ ok: true, run_id, lookbackDays, summary: agg.summary, perTicker: perTicker.map((r: any) => ({ ticker: r.ticker, ...r.summary })), equityCurve: agg.equityCurve, tradesCount: flatTrades.length })
  } catch (e: any) {
    return json({ error: e?.message || String(e) }, 500)
  }
}

