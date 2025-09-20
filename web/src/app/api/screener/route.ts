import yahooFinance from 'yahoo-finance2'
import Parser from 'rss-parser'
// vader-sentiment types are not strict; import as any
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { SentimentIntensityAnalyzer: VADER } = require('vader-sentiment')
import { sma, rsi14, atr14 } from '../../../lib/indicators'
import buyCfg from '../../../config/buyzones.json'


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

  // Benchmark (SPY) for relative strength metrics — fetch once
  let benchMap: Record<string, number> = {}
  try {
    const period1 = new Date(Date.now() - 1000 * 60 * 60 * 24 * 365 * 2)
    const chBench: any = await (yahooFinance as any).chart('SPY', { period1, interval: '1d' } as any)
    let histB: any[] = Array.isArray(chBench?.quotes) && chBench.quotes.length ? chBench.quotes : []
    if (!histB.length) {
      const tsB: number[] = chBench?.timestamp || []
      const qB = chBench?.indicators?.quote?.[0] || {}
      histB = tsB.map((tsVal: number, i: number) => ({
        date: new Date(tsVal * 1000),
        close: qB?.close?.[i],
      })).filter((x: any) => Number.isFinite(x.close))
    }
    benchMap = Object.fromEntries(
      histB.map((x: any) => [ (x.date ? new Date(x.date).toISOString().slice(0, 10) : ''), Number(x.close) ])
        .filter(([d, v]: any[]) => d && Number.isFinite(v))
    )
  } catch {
    // best-effort; RS metrics will be null if SPY fetch fails
    benchMap = {}
  }

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
          volume: q?.volume?.[i],
        })).filter(x => Number.isFinite(x.close) && Number.isFinite(x.high) && Number.isFinite(x.low))
      }
      if (!hist?.length) throw new Error('no price data')
      const close = hist.map(x => Number(x.close))
      const high = hist.map(x => Number(x.high))
      const low = hist.map(x => Number(x.low))
      const volume = hist.map(x => Number(x.volume ?? 0))
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

      // Volume trend (20d MA vs 5 days ago)
      const vol20 = sma(volume, 20)
      const volTrendUp = (vol20[last] != null && vol20[last - 5] != null) ? (vol20[last]! > vol20[last - 5]!) : null

      // 52w position (approx 252 trading days)
      const look = Math.min(252, close.length)
      const win = close.slice(close.length - look)
      const high52w = win.length ? Math.max(...win) : null
      const low52w = win.length ? Math.min(...win) : null
      const distToHighPct = (price != null && high52w != null && high52w > 0) ? ((high52w - price) / high52w) * 100 : null
      const distToLowPct = (price != null && low52w != null && low52w > 0) ? ((price - low52w) / low52w) * 100 : null

      // Relative strength vs SPY
      let rsRatio: number | null = null
      let rsSlope30: number | null = null
      {
        const d = dates[last]
        const spyP = d ? benchMap[d] : undefined
        if (price != null && Number.isFinite(spyP)) rsRatio = price / spyP!
        const N = 30
        const rsArr: number[] = []
        for (let i = Math.max(0, close.length - N); i < close.length; i++) {
          const d2 = dates[i]
          const sp = d2 ? benchMap[d2] : undefined
          const c = close[i]
          if (Number.isFinite(c) && Number.isFinite(sp)) rsArr.push(c / (sp as number))
        }
        if (rsArr.length >= 5) rsSlope30 = (rsArr[rsArr.length - 1] - rsArr[0]) / rsArr.length
      }

      // Fundamentals (best-effort) – fetch in two passes and be tolerant to API/type shape
      const sumA = await yahooFinance.quoteSummary(t, { modules: ['financialData', 'price'] as any }).catch(() => null)
      const sumB = await yahooFinance.quoteSummary(t, { modules: ['defaultKeyStatistics', 'majorHoldersBreakdown', 'summaryDetail'] as any }).catch(() => null)
      const sumC = await yahooFinance.quoteSummary(t, { modules: ['assetProfile', 'summaryProfile'] as any }).catch(() => null)
      const qs: any = { ...(sumA as any || {}), ...(sumB as any || {}), ...(sumC as any || {}) }
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
      // Additional valuation metrics (best-effort; may be null)
      const trailingPE = safeNum(qs?.summaryDetail?.trailingPE?.raw ?? qs?.summaryDetail?.trailingPE ?? qs?.defaultKeyStatistics?.trailingPE?.raw ?? qs?.defaultKeyStatistics?.trailingPE)
      const forwardPE = safeNum(qs?.summaryDetail?.forwardPE?.raw ?? qs?.summaryDetail?.forwardPE ?? qs?.defaultKeyStatistics?.forwardPE?.raw ?? qs?.defaultKeyStatistics?.forwardPE)
      const peg = safeNum(qs?.defaultKeyStatistics?.pegRatio?.raw ?? qs?.defaultKeyStatistics?.pegRatio)
      const p_s = safeNum(qs?.summaryDetail?.priceToSalesTrailing12Months?.raw ?? qs?.summaryDetail?.priceToSalesTrailing12Months)
      const p_b = safeNum(qs?.defaultKeyStatistics?.priceToBook?.raw ?? qs?.defaultKeyStatistics?.priceToBook ?? qs?.summaryDetail?.priceToBook?.raw ?? qs?.summaryDetail?.priceToBook)
      const ev_to_ebitda = safeNum(qs?.defaultKeyStatistics?.enterpriseToEbitda?.raw ?? qs?.defaultKeyStatistics?.enterpriseToEbitda)

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

      const sector = (qs?.assetProfile?.sector ?? qs?.summaryProfile?.sector) || null
      const industry = (qs?.assetProfile?.industry ?? qs?.summaryProfile?.industry) || null

      const fundamentals = {
        fcf_yield: fcfYield,
        nd_to_ebitda: ndToEbitda,
        gross_margin: grossMargin,
        revenue_growth: revenueGrowth,
        insider_own: insiderOwn,
        pe: trailingPE,
        fwd_pe: forwardPE,
        peg,
        p_s,
        p_b,
        ev_to_ebitda,
        sector,
        industry,
      }

      const flags: string[] = []
      const allNull = [fcfYield, ndToEbitda, grossMargin, revenueGrowth, insiderOwn].every(v => v == null)
      if (allNull) flags.push('fundamentals_missing')

      // Phase 2 – compute buy zones, exit levels, and position health
      const lastAtr = (atr[last] != null ? atr[last]! : null)
      const ma20v = sma20[last]
      const ma50v = sma50[last]
      const ma200v = sma200[last]
      const cfg: any = (buyCfg as any) || {}

      const buy_zones: Array<{ type: string; ma?: string; price_low: number; price_high: number; confidence?: number; rationale?: string }> = []
      // SMA20 pullback zone: [MA20 - k*ATR, MA20]
      if (cfg?.sma20?.enabled && typeof ma20v === 'number' && typeof lastAtr === 'number') {
        const k = Number(cfg.sma20.k_atr ?? 1)
        const lo = ma20v - k * lastAtr
        const hi = ma20v
        if (Number.isFinite(lo) && Number.isFinite(hi)) {
          buy_zones.push({ type: 'sma_pullback', ma: 'SMA20', price_low: lo, price_high: hi, confidence: 0.6, rationale: `Zone = SMA20 − ${k}·ATR → SMA20` })
        }
      }
      // SMA50 pullback zone: [MA50 - k*ATR, MA50]
      if (cfg?.sma50?.enabled && typeof ma50v === 'number' && typeof lastAtr === 'number') {
        const k = Number(cfg.sma50.k_atr ?? 1)
        const lo = ma50v - k * lastAtr
        const hi = ma50v
        if (Number.isFinite(lo) && Number.isFinite(hi)) {
          buy_zones.push({ type: 'sma_pullback', ma: 'SMA50', price_low: lo, price_high: hi, confidence: 0.6, rationale: `Zone = SMA50 − ${k}·ATR → SMA50` })
        }
      }
      // Breakout retest zone around recent high
      {
        const lb = Math.max(20, Math.min(120, Number(cfg?.breakout?.lookback ?? 60)))
        if (typeof lastAtr === 'number' && close.length >= lb) {
          const slice = close.slice(-lb)
          const rHigh = Math.max(...slice)
          if (Number.isFinite(rHigh)) {
            const k = Number(cfg?.breakout?.k_atr ?? 0.5)
            const lo = rHigh - k * lastAtr
            const hi = rHigh + k * lastAtr
            if (Number.isFinite(lo) && Number.isFinite(hi)) {
              buy_zones.push({ type: 'breakout_retest', price_low: lo, price_high: hi, confidence: 0.5, rationale: `Retest zone near ${lb}d high ± ${k}·ATR` })
            }
          }
        }
      }

      // Exit levels
      const stopMaName = String(cfg?.stops?.ma || 'sma50').toLowerCase()
      const stopBase = stopMaName === 'sma20' ? ma20v : ma50v
      const stop_suggest = (typeof stopBase === 'number' && typeof lastAtr === 'number') ? (stopBase - Number(cfg?.stops?.atr_mult ?? 1) * lastAtr) : null
      const multiples: number[] = Array.isArray(cfg?.targets?.multiples) ? (cfg.targets.multiples as number[]) : [1.5, 2.5]
      const targets = (typeof price === 'number' && typeof stop_suggest === 'number' && price > stop_suggest)
        ? multiples.map(m => price + m * (price - stop_suggest)).filter(x => Number.isFinite(x))
        : []
      const exit_levels = { stop_suggest, targets }

      // Position health
      function inZone(p?: number | null) {
        if (typeof p !== 'number') return false
        return buy_zones.some(z => p >= z.price_low && p <= z.price_high)
      }
      const in_buy_zone = inZone(price)
      const dist_to_stop_pct = (typeof price === 'number' && typeof stop_suggest === 'number' && price > 0) ? Number((((price - stop_suggest) / price) * 100).toFixed(2)) : null
      const t1 = targets[0]
      const dist_to_t1_pct = (typeof price === 'number' && typeof t1 === 'number' && price > 0) ? Number((((t1 - price) / price) * 100).toFixed(2)) : null

      // Entry readiness: base 30, +40 if in zone, +20 if trendUp, +10 if RSI in 45-65
      let entry_readiness = 30
      if (in_buy_zone) entry_readiness += 40
      if (trendUp) entry_readiness += 20
      if (typeof rsi[last] === 'number' && rsi[last]! >= 45 && rsi[last]! <= 65) entry_readiness += 10
      entry_readiness = Math.max(0, Math.min(100, Math.round(entry_readiness)))

      // Exit risk: base 20; +30 if price < SMA50; +30 if price < SMA200; +20 if dist_to_stop_pct < 5
      let exit_risk = 20
      if (typeof price === 'number' && typeof ma50v === 'number' && price < ma50v) exit_risk += 30
      if (typeof price === 'number' && typeof ma200v === 'number' && price < ma200v) exit_risk += 30
      if (typeof dist_to_stop_pct === 'number' && dist_to_stop_pct < 5) exit_risk += 20
      exit_risk = Math.max(0, Math.min(100, Math.round(exit_risk)))

      const position_health = { entry_readiness, exit_risk, in_buy_zone, dist_to_stop_pct, dist_to_t1_pct }

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
        technicals: { provider: 'yahoo', rsi: rsi[last], atr_pct: atrPct, sma20: sma20[last], sma50: sma50[last], sma200: sma200[last], vol_ma20: vol20[last], vol_trend_up: volTrendUp, rs_ratio: rsRatio, rs_slope_30: rsSlope30, dist_52w_high_pct: distToHighPct, dist_52w_low_pct: distToLowPct },
        sentiment: { mean7, count7: last7.length },
        buy_zones,
        exit_levels,
        position_health,
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

