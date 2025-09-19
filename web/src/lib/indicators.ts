export function sma(values: number[], period: number) {
  if (!Array.isArray(values) || period <= 0) return Array(values?.length || 0).fill(null)
  if (values.length < period) return Array(values.length).fill(null)
  const out = Array(values.length).fill(null) as (number|null)[]
  let sum = 0
  for (let i = 0; i < values.length; i++) {
    sum += values[i]
    if (i >= period) sum -= values[i - period]
    if (i >= period - 1) out[i] = sum / period
  }
  return out
}

export function rsi14(closes: number[], period = 14) {
  const n = closes.length
  if (n < period + 1) return Array(n).fill(null)
  const rsi = Array(n).fill(null) as (number|null)[]
  let gains = 0, losses = 0
  for (let i = 1; i <= period; i++) {
    const ch = closes[i] - closes[i - 1]
    if (ch >= 0) gains += ch; else losses -= ch
  }
  let avgG = gains / period, avgL = losses / period
  rsi[period] = 100 - 100 / (1 + (avgG / (avgL || 1e-9)))
  for (let i = period + 1; i < n; i++) {
    const ch = closes[i] - closes[i - 1]
    const g = ch > 0 ? ch : 0
    const l = ch < 0 ? -ch : 0
    avgG = (avgG * (period - 1) + g) / period
    avgL = (avgL * (period - 1) + l) / period
    rsi[i] = 100 - 100 / (1 + (avgG / (avgL || 1e-9)))
  }
  return rsi
}

export function atr14(high: number[], low: number[], close: number[], period = 14) {
  const n = close.length
  const tr: (number|null)[] = Array(n).fill(null)
  for (let i = 0; i < n; i++) {
    if (i === 0) { tr[i] = (high[i] - low[i]); continue }
    const hl = high[i] - low[i]
    const hc = Math.abs(high[i] - close[i-1])
    const lc = Math.abs(low[i] - close[i-1])
    tr[i] = Math.max(hl, hc, lc)
  }
  const atr = Array(n).fill(null) as (number|null)[]
  let sum = 0
  for (let i = 0; i < n; i++) {
    sum += (tr[i] ?? 0)
    if (i === period) atr[i] = sum / period
    else if (i > period) atr[i] = ((atr[i-1] as number) * (period - 1) + (tr[i] as number)) / period
  }
  return atr
}

