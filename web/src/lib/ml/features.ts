import type { LinearModel } from './model'

export type BaseFeatures = Record<string, number>

export function toBaseFeatures(x: any): BaseFeatures {
  const f: BaseFeatures = {}
  const price = num(x?.price)
  const sma20 = num(x?.technicals?.sma20)
  const sma50 = num(x?.technicals?.sma50)
  const sma200 = num(x?.technicals?.sma200)
  const rsi = num(x?.technicals?.rsi)
  const atrPct = num(x?.technicals?.atr_pct)

  f.price_over_sma20 = (price != null && sma20 != null && sma20 !== 0) ? price / sma20 : 1
  f.price_over_sma50 = (price != null && sma50 != null && sma50 !== 0) ? price / sma50 : 1
  f.price_over_sma200 = (price != null && sma200 != null && sma200 !== 0) ? price / sma200 : 1
  f.rsi_norm = (rsi != null ? rsi : 50) / 100
  if (atrPct != null) f.atr_pct = atrPct
  // ATR buckets: [-inf,1),[1,2),[2,4),[4,8),[8,inf)
  const b = bucket(atrPct, [-Infinity, 1, 2, 4, 8, Infinity])
  for (let i = 0; i < 5; i++) f[`atr_bucket_${i}`] = (b === i ? 1 : 0)

  // Fundamental & sentiment examples (if present)
  const fcf_yield = num(x?.fundamentals?.fcf_yield)
  const nd_to_ebitda = num(x?.fundamentals?.nd_to_ebitda)
  const rev_g = num(x?.fundamentals?.revenue_growth)
  const gm = num(x?.fundamentals?.gross_margin)
  if (fcf_yield != null) f.fcf_yield = fcf_yield
  if (nd_to_ebitda != null) f.nd_to_ebitda = nd_to_ebitda
  if (rev_g != null) f.revenue_growth = rev_g
  if (gm != null) f.gross_margin = gm

  const sent7 = num(x?.sentiment?.mean7)
  if (sent7 != null) f.sent_mean7 = sent7

  // Enhanced technical features
  const vol20_rising = x?.technicals?.vol20_rising
  const price_gt_ma20 = x?.technicals?.price_gt_ma20
  if (vol20_rising != null) f.vol20_rising = vol20_rising ? 1 : 0
  if (price_gt_ma20 != null) f.price_gt_ma20 = price_gt_ma20 ? 1 : 0

  // Momentum features
  if (rsi != null) {
    f.rsi_oversold = rsi < 30 ? 1 : 0
    f.rsi_overbought = rsi > 70 ? 1 : 0
  }

  // Price position features
  if (price != null && sma20 != null && sma50 != null && sma200 != null) {
    f.sma_alignment = (sma20 > sma50 && sma50 > sma200) ? 1 : 0
    f.above_all_smas = (price > sma20 && price > sma50 && price > sma200) ? 1 : 0
  }

  return f
}

export function vectorForModel(m: LinearModel, f: BaseFeatures): number[] {
  const out: number[] = []
  for (let i = 0; i < m.features.length; i++) {
    const k = m.features[i]
    let v = f[k]
    if (v == null || !Number.isFinite(v)) v = 0
    const mean = m.norm?.mean?.[k]
    const std = m.norm?.std?.[k]
    if (Number.isFinite(mean) && Number.isFinite(std) && std && std > 0) v = (v - (mean as number)) / (std as number)
    out.push(v)
  }
  return out
}

export function dot(m: LinearModel, x: number[]): number {
  let s = m.intercept
  for (let i = 0; i < m.coef.length && i < x.length; i++) s += m.coef[i] * x[i]
  return s
}

function num(x: any): number | null {
  const n = Number(x)
  return Number.isFinite(n) ? n : null
}

function bucket(v: number | null, edges: number[]): number | null {
  if (v == null || !Number.isFinite(v)) return null
  for (let i = 0; i < edges.length - 1; i++) {
    if (v >= edges[i] && v < edges[i + 1]) return i
  }
  return null
}



export type FeatureContrib = { key: string; value: number; weight: number; mean?: number; std?: number; normValue: number; contrib: number }

export function featureContributions(m: LinearModel, f: BaseFeatures): FeatureContrib[] {
  const out: FeatureContrib[] = []
  for (let i = 0; i < m.features.length; i++) {
    const k = m.features[i]
    const w = m.coef[i] ?? 0
    let v = f[k]
    if (v == null || !Number.isFinite(v)) v = 0
    const mean = m.norm?.mean?.[k]
    const std = m.norm?.std?.[k]
    let vn = v
    if (Number.isFinite(mean) && Number.isFinite(std) && (std as number) > 0) {
      vn = (v - (mean as number)) / (std as number)
    }
    out.push({ key: k, value: v, weight: w, mean, std, normValue: vn, contrib: w * vn })
  }
  return out
}
