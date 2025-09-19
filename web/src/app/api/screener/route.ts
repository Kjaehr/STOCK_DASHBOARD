import yahooFinance from 'yahoo-finance2'
import Parser from 'rss-parser'
// vader-sentiment types are not strict; import as any
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { SentimentIntensityAnalyzer: VADER } = require('vader-sentiment')
import { sma, rsi14, atr14 } from '../../../lib/indicators'

export const runtime = 'nodejs'
export const revalidate = 900 // 15 min route cache window (advisory)
;(yahooFinance as any).suppressNotices && (yahooFinance as any).suppressNotices(['ripHistorical'])


function bypass(req: Request): boolean {
  try {
    const u = new URL(req.url)
    return u.searchParams.get('refresh') === '1' || u.searchParams.has('t')
  } catch {
    return false
  }
}

async function pLimit<T>(n: number, tasks: (() => Promise<T>)[]) {
  const out: T[] = new Array(tasks.length)
  let i = 0
  async function run() {
    for (;;) {
      const idx = i++
      if (idx >= tasks.length) break
      out[idx] = await tasks[idx]()
    }
  }
  const workers = Array.from({ length: Math.min(n, tasks.length) }, () => run())
  await Promise.all(workers)
  return out
}

function safeNum(x: any): number | null { const n = Number(x); return Number.isFinite(n) ? n : null }

export async function GET(req: Request) {
  const url = new URL(req.url)
  const raw = (url.searchParams.get('tickers') || '').split(',').map(s => s.trim()).filter(Boolean)
  if (!raw.length) return new Response(JSON.stringify({ error: 'tickers required' }), { status: 400, headers: { 'content-type': 'application/json' } })

  const tickers = raw.slice(0, 10) // rate-limit friendly: max 10 per request
  const noStore = bypass(req)

  const parser = new Parser()

  async function fetchOne(t: string) {
    try {
      const period1 = new Date(Date.now() - 1000 * 60 * 60 * 24 * 365 * 2)
      const ch: any = await yahooFinance.chart(t, { period1, interval: '1d' } as any)
      let hist: any[] = Array.isArray(ch?.quotes) && ch.quotes.length ? ch.quotes : []
      if (!hist.length) {
        const ts: number[] = ch?.timestamp || []
        const q = ch?.indicators?.quote?.[0] || {}
        hist = ts.map((tsVal: number, i: number) => ({
          date: new Date(tsVal * 1000),
          close: q?.close?.[i],
          high: q?.high?.[i],
          low: q?.low?.[i],
        })).filter(x => Number.isFinite(x.close) && Number.isFinite(x.high) && Number.isFinite(x.low))
      }
      if (!hist?.length) throw new Error('no price data')
      const close = hist.map(x => Number(x.close))
      const high = hist.map(x => Number(x.high))
      const low = hist.map(x => Number(x.low))
      const dates = hist.map(x => (x.date ? new Date(x.date).toISOString().slice(0, 10) : undefined))
      const last = close.length - 1
      const price = close[last]

      const sma20 = sma(close, 20)
      const sma50 = sma(close, 50)
      const sma200 = sma(close, 200)
      const rsi = rsi14(close, 14)
      const atr = atr14(high, low, close, 14)
      const atrPct = atr[last] != null && price ? (atr[last]! / price) * 100 : null
      const trendUp = (sma50[last] != null && sma200[last] != null && price != null) ? ((sma50[last]! > sma200[last]!) && (price > sma200[last]!)) : false

      // Fundamentals (best-effort) – fetch in two passes and be tolerant to API/type shape
      const sumA = await yahooFinance.quoteSummary(t, { modules: ['financialData', 'price'] as any }).catch(() => null)
      const sumB = await yahooFinance.quoteSummary(t, { modules: ['defaultKeyStatistics', 'majorHoldersBreakdown', 'summaryDetail'] as any }).catch(() => null)
      const qs: any = { ...(sumA as any || {}), ...(sumB as any || {}) }
      const mcap = safeNum(qs?.price?.marketCap?.raw ?? qs?.price?.marketCap)
      const fcf = safeNum(qs?.financialData?.freeCashflow?.raw ?? qs?.financialData?.freeCashflow ?? qs?.defaultKeyStatistics?.freeCashflow?.raw ?? qs?.defaultKeyStatistics?.freeCashflow)
      const ebitda = safeNum(qs?.financialData?.ebitda?.raw ?? qs?.financialData?.ebitda)
      const totalDebt = safeNum(qs?.financialData?.totalDebt?.raw ?? qs?.financialData?.totalDebt ?? qs?.defaultKeyStatistics?.totalDebt?.raw ?? qs?.defaultKeyStatistics?.totalDebt)
      const totalCash = safeNum(qs?.financialData?.totalCash?.raw ?? qs?.financialData?.totalCash)
      const netDebt = (totalDebt != null && totalCash != null) ? (totalDebt - totalCash) : (totalDebt != null ? totalDebt : null)
      const ndToEbitda = (netDebt != null && ebitda != null && ebitda !== 0) ? (netDebt / ebitda) : null
      const grossMargin = safeNum(qs?.financialData?.grossMargins?.raw ?? qs?.financialData?.grossMargins ?? qs?.summaryDetail?.grossMargins?.raw ?? qs?.summaryDetail?.grossMargins)
      const revenueGrowth = safeNum(qs?.financialData?.revenueGrowth?.raw ?? qs?.financialData?.revenueGrowth)
      const insiderOwn = safeNum(qs?.defaultKeyStatistics?.heldPercentInsiders?.raw ?? qs?.defaultKeyStatistics?.heldPercentInsiders ?? qs?.majorHoldersBreakdown?.insidersPercentHeld?.raw ?? qs?.majorHoldersBreakdown?.insidersPercentHeld)
      const fcfYield = (fcf != null && mcap != null && mcap > 0) ? (fcf / mcap) : null

      function scoreFundamentals() {
        let pts = 0
        if (fcfYield != null) { if (fcfYield > 0.08) pts += 12; else if (fcfYield >= 0.04) pts += 8; else if (fcfYield >= 0) pts += 4 }
        if (ndToEbitda != null) { if (ndToEbitda < 1) pts += 8; else if (ndToEbitda < 2) pts += 5; else if (ndToEbitda < 3) pts += 2 }
        if (grossMargin != null) { if (grossMargin > 0.45) pts += 8; else if (grossMargin >= 0.30) pts += 4 }
        if (revenueGrowth != null) { if (revenueGrowth > 0.15) pts += 6; else if (revenueGrowth >= 0.05) pts += 3; else if (revenueGrowth >= 0) pts += 1 }
        if (insiderOwn != null) { if (insiderOwn >= 0.10) pts += 6; else if (insiderOwn >= 0.03) pts += 3 }
        return pts
      }
      const fundPoints = scoreFundamentals()

      // News + VADER
      const rss = await parser.parseURL(`https://news.google.com/rss/search?q=${encodeURIComponent(t + ' stock')}&hl=en-US&gl=US&ceid=US:en`).catch(() => ({ items: [] as any[] }))
      const last7 = (rss.items || []).filter(i => {
        const d = i.isoDate ? new Date(i.isoDate) : (i.pubDate ? new Date(i.pubDate) : null)
        return d && (Date.now() - d.getTime()) / 86400000 <= 7
      })
      const scores: number[] = last7.map(i => (VADER.polarity_scores(String(i.title || '')).compound as number) || 0)
      const mean7 = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0

      // Simple scoring aligned with weights 0.40/0.35/0.25
      const techPoints = (trendUp ? 20 : 0) + ((rsi[last] != null && rsi[last]! >= 45 && rsi[last]! <= 65) ? 10 : 0) + ((atrPct != null && atrPct < 5) ? 5 : 0)
      const sentPoints = (mean7 > 0.1 ? 10 : (mean7 < -0.1 ? 0 : 5)) + ((last7.length > 5) ? 5 : 0)
      const score = Math.round(0.40 * fundPoints + 0.35 * techPoints + 0.25 * sentPoints)

      const fundamentals = {
        fcf_yield: fcfYield,
        nd_to_ebitda: ndToEbitda,
        gross_margin: grossMargin,
        revenue_growth: revenueGrowth,
        insider_own: insiderOwn,
      }

      const flags: string[] = []
      const allNull = [fcfYield, ndToEbitda, grossMargin, revenueGrowth, insiderOwn].every(v => v == null)
      if (allNull) flags.push('fundamentals_missing')

      return {
        ticker: t,
        price,
        sma50: sma50[last],
        sma200: sma200[last],
        updated_at: dates[last] || new Date().toISOString().slice(0, 10),
        score,
        fund_points: fundPoints,
        tech_points: techPoints,
        sent_points: sentPoints,
        flags,
        fundamentals,
        technicals: { provider: 'yahoo', rsi: rsi[last], atr_pct: atrPct, sma50: sma50[last], sma200: sma200[last] },
        sentiment: { mean7, count7: last7.length },
      }
    } catch (e: any) {
      return { ticker: t, error: e?.message || String(e), flags: ['fetch_fail'] }
    }
  }

  const tasks = tickers.map(t => () => fetchOne(t))
  const items = await pLimit(5, tasks)

  const headers: Record<string, string> = { 'content-type': 'application/json' }
  if (noStore) headers['cache-control'] = 'no-store'
  else headers['cache-control'] = 'public, s-maxage=900, stale-while-revalidate=900'

  return new Response(JSON.stringify({ generated_at: new Date().toISOString(), items }), { headers })
}

