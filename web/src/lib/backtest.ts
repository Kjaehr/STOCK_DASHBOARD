import { sma, rsi14, atr14 } from './indicators'

export type StrategyParams = {
  sma20?: { k_atr?: number; enabled?: boolean }
  sma50?: { k_atr?: number; enabled?: boolean }
  breakout?: { lookback?: number; k_atr?: number; enabled?: boolean }
  stops?: { ma?: 'sma20'|'sma50'; atr_mult?: number }
  targets?: { multiples?: number[] }
  filters?: { trendUp?: boolean; rsiRange?: [number, number] }
}

export type Trade = {
  ticker: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  r: number
  pnl_pct: number
  bars_held: number
  exit_reason: 'target'|'stop'|'timeout'
}

export type BacktestSummary = {
  trades: number
  wins: number
  losses: number
  hit_rate: number
  avg_win_R: number | null
  avg_loss_R: number | null
  expectancy_R: number | null
}

export type BacktestResult = {
  ticker: string
  summary: BacktestSummary
  equityCurve: { i: number; eq: number }[] // cumulative R per trade index
  trades: Trade[]
}

export type OHLC = { date: string; close: number; high: number; low: number }

function defaultParams(): Required<StrategyParams> {
  return {
    sma20: { enabled: true, k_atr: 1 },
    sma50: { enabled: true, k_atr: 1 },
    breakout: { enabled: true, lookback: 60, k_atr: 0.5 },
    stops: { ma: 'sma50', atr_mult: 1 },
    targets: { multiples: [1.5, 2.5] },
    filters: { trendUp: true, rsiRange: [45, 65] },
  }
}

function inRange(x: number | null | undefined, a: number, b: number) {
  if (typeof x !== 'number') return false
  return x >= Math.min(a, b) && x <= Math.max(a, b)
}

export function backtestOne(ticker: string, ohlc: OHLC[], paramsIn?: StrategyParams): BacktestResult {
  const params = { ...defaultParams(), ...(paramsIn || {}) }
  const close = ohlc.map(o => o.close)
  const high = ohlc.map(o => o.high)
  const low = ohlc.map(o => o.low)
  const dates = ohlc.map(o => o.date)
  const sma20 = sma(close, 20)
  const sma50 = sma(close, 50)
  const sma200 = sma(close, 200)
  const rsi = rsi14(close, 14)
  const atr = atr14(high, low, close, 14)

  const trades: Trade[] = []
  let eq = 0
  const equityCurve: { i: number; eq: number }[] = []

  let i = 200 // need enough warmup
  while (i < close.length - 1) {
    const price = close[i]
    const d = dates[i]
    const a = atr[i] as number | null
    const ma20 = sma20[i] as number | null
    const ma50 = sma50[i] as number | null
    const ma200 = sma200[i] as number | null
    if (!(Number.isFinite(price) && Number.isFinite(a ?? NaN))) { i++; continue }

    // Filters
    if (params.filters?.trendUp) {
      const trendUp = (ma50 != null && ma200 != null && price != null) ? ((ma50! > ma200!) && (price > ma200!)) : false
      if (!trendUp) { i++; continue }
    }
    if (params.filters?.rsiRange) {
      const r = rsi[i]
      const [lo, hi] = params.filters.rsiRange
      if (!(typeof r === 'number' && r >= lo && r <= hi)) { i++; continue }
    }

    // Build buy zones for day i
    const zones: Array<{ low: number; high: number }> = []
    if (params.sma20?.enabled && typeof ma20 === 'number' && typeof a === 'number') {
      zones.push({ low: ma20 - (params.sma20.k_atr ?? 1) * a, high: ma20 })
    }
    if (params.sma50?.enabled && typeof ma50 === 'number' && typeof a === 'number') {
      zones.push({ low: ma50 - (params.sma50.k_atr ?? 1) * a, high: ma50 })
    }
    if (params.breakout?.enabled && typeof a === 'number') {
      const look = Math.max(20, Math.min(180, params.breakout.lookback ?? 60))
      if (i >= look) {
        const recent = close.slice(i - look + 1, i + 1)
        const rHigh = Math.max(...recent)
        const k = params.breakout.k_atr ?? 0.5
        zones.push({ low: rHigh - k * a, high: rHigh + k * a })
      }
    }

    // Entry if price is inside any zone
    const inZone = zones.some(z => inRange(price, z.low, z.high))
    if (!inZone) { i++; continue }

    // Open trade at close[i]
    const entry = price
    const stopMa = (params.stops?.ma === 'sma20') ? ma20 : ma50
    const stop = (typeof stopMa === 'number' && typeof a === 'number') ? (stopMa - (params.stops?.atr_mult ?? 1) * a) : null
    if (!(typeof stop === 'number') || !(entry > stop)) { i++; continue }
    const Rden = entry - stop

    // Targets computed off entry/stop distance
    const multiples = (params.targets?.multiples && params.targets.multiples.length ? params.targets.multiples : [1.5, 2.5]).slice().sort((x, y) => x - y)
    const t1 = entry + (multiples[0] ?? 1.5) * Rden

    // Walk forward until exit
    let j = i + 1
    let exit = close[j]
    let reason: Trade['exit_reason'] = 'timeout'
    while (j < close.length) {
      const p = close[j]
      if (typeof p !== 'number') { j++; continue }
      if (p <= stop) { exit = p; reason = 'stop'; break }
      if (p >= t1) { exit = p; reason = 'target'; break }
      j++
    }
    const bars = j - i
    const pnlPct = ((exit - entry) / entry) * 100
    const rVal = (exit - entry) / Rden
    trades.push({ ticker, entry_date: d, exit_date: dates[Math.min(j, dates.length - 1)], entry_price: Number(entry.toFixed(4)), exit_price: Number(exit.toFixed(4)), r: Number(rVal.toFixed(2)), pnl_pct: Number(pnlPct.toFixed(2)), bars_held: bars, exit_reason: reason })
    eq += rVal
    equityCurve.push({ i: trades.length, eq: Number(eq.toFixed(2)) })

    // Move past exit bar to avoid overlapping trades
    i = Math.max(j, i + 1)
  }

  // Summary
  const winsArr = trades.filter(t => t.r > 0).map(t => t.r)
  const lossArr = trades.filter(t => t.r <= 0).map(t => t.r)
  const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null
  const hit_rate = trades.length ? winsArr.length / trades.length : 0
  const avg_win_R = avg(winsArr)
  const avg_loss_R = avg(lossArr)
  const expectancy_R = trades.length ? Number(((winsArr.reduce((a,b)=>a+b,0) + lossArr.reduce((a,b)=>a+b,0)) / trades.length).toFixed(2)) : null

  const summary: BacktestSummary = {
    trades: trades.length,
    wins: winsArr.length,
    losses: lossArr.length,
    hit_rate: Number(hit_rate.toFixed(2)),
    avg_win_R: (avg_win_R != null ? Number(avg_win_R.toFixed(2)) : null),
    avg_loss_R: (avg_loss_R != null ? Number(avg_loss_R.toFixed(2)) : null),
    expectancy_R,
  }

  return { ticker, summary, equityCurve, trades }
}

export function aggregateResults(results: BacktestResult[]) {
  const totalTrades = results.reduce((a, r) => a + r.summary.trades, 0)
  const totalWins = results.reduce((a, r) => a + r.summary.wins, 0)
  const totalLosses = results.reduce((a, r) => a + r.summary.losses, 0)
  const allR = results.flatMap(r => r.trades.map(t => t.r))
  const winsR = allR.filter(x => x > 0)
  const lossR = allR.filter(x => x <= 0)
  const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null
  const hit_rate = totalTrades ? totalWins / totalTrades : 0
  const avg_win_R = avg(winsR)
  const avg_loss_R = avg(lossR)
  const expectancy_R = totalTrades ? Number(((allR.reduce((a,b)=>a+b,0)) / totalTrades).toFixed(2)) : null

  // Equity curve over trade index (concatenate in ticker order)
  let eq = 0
  const equityCurve: { i: number; eq: number }[] = []
  results.forEach(r => {
    r.trades.forEach(t => { eq += t.r; equityCurve.push({ i: equityCurve.length + 1, eq: Number(eq.toFixed(2)) }) })
  })

  return {
    summary: {
      trades: totalTrades,
      wins: totalWins,
      losses: totalLosses,
      hit_rate: Number(hit_rate.toFixed(2)),
      avg_win_R: (avg_win_R != null ? Number(avg_win_R.toFixed(2)) : null),
      avg_loss_R: (avg_loss_R != null ? Number(avg_loss_R.toFixed(2)) : null),
      expectancy_R,
    } as BacktestSummary,
    equityCurve,
  }
}

