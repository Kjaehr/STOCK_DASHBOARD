"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockData, StockMeta } from '../../../types'
import { PriceChart, LineChart } from '../../../components/Charts'
import { BASE, DATA_BASE } from '../../../base'

const META_CACHE_KEY = 'stockdash:meta'
const TICKER_CACHE_PREFIX = 'stockdash:ticker:'
const CACHE_TTL_MS = 10 * 60 * 1000

type CacheRecord<T> = {
  ts: number
  payload: T
}

type Eval = {
  points: number
  max: number
  hint: string
}

type VolumeEval = Eval & { value: string }
type FlowEval = Eval & { value: string }
type SignalEval = Eval & { value: string }

function readCache<T>(key: string): CacheRecord<T> | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = window.localStorage.getItem(key)
    if (!raw) return null
    return JSON.parse(raw) as CacheRecord<T>
  } catch {
    return null
  }
}

function writeCache<T>(key: string, payload: T) {
  if (typeof window === 'undefined') return
  try {
    const record: CacheRecord<T> = { ts: Date.now(), payload }
    window.localStorage.setItem(key, JSON.stringify(record))
  } catch (e) {
    console.warn('cache write failed', e)
  }
}

function isFresh(record: CacheRecord<unknown> | null) {
  return !!record && Date.now() - record.ts <= CACHE_TTL_MS
}

function tickerCacheKey(id: string) {
  return `${TICKER_CACHE_PREFIX}${id}`
}

function googleNewsUrl(id: string) {
  const query = `${id} stock`
  const params = new URLSearchParams({ q: query, hl: 'en-US', gl: 'US', ceid: 'US:en' })
  return `https://news.google.com/search?${params.toString()}`
}

function evalFcfYield(v?: number | null): Eval {
  if (v == null) return { points: 0, max: 12, hint: 'Mangler FCF data' }
  if (v > 0.08) return { points: 12, max: 12, hint: 'FCF yield > 8% (+12)' }
  if (v >= 0.04) return { points: 8, max: 12, hint: 'FCF yield mellem 4% og 8% (+8)' }
  if (v >= 0) return { points: 4, max: 12, hint: 'FCF yield mellem 0% og 4% (+4)' }
  return { points: 0, max: 12, hint: 'FCF yield < 0% (+0)' }
}

function evalNetDebt(v?: number | null): Eval {
  if (v == null) return { points: 0, max: 8, hint: 'Mangler Net Debt / EBITDA' }
  if (v < 1) return { points: 8, max: 8, hint: 'Net Debt / EBITDA < 1 (+8)' }
  if (v < 2) return { points: 5, max: 8, hint: 'Net Debt / EBITDA mellem 1 og 2 (+5)' }
  if (v < 3) return { points: 2, max: 8, hint: 'Net Debt / EBITDA mellem 2 og 3 (+2)' }
  return { points: 0, max: 8, hint: 'Net Debt / EBITDA > 3 (+0)' }
}

function evalGrossMargin(v?: number | null): Eval {
  if (v == null) return { points: 0, max: 8, hint: 'Mangler gross margin' }
  if (v > 0.45) return { points: 8, max: 8, hint: 'Gross margin > 45% (+8)' }
  if (v >= 0.3) return { points: 4, max: 8, hint: 'Gross margin mellem 30% og 45% (+4)' }
  return { points: 0, max: 8, hint: 'Gross margin < 30% (+0)' }
}

function evalRevenueGrowth(v?: number | null): Eval {
  if (v == null) return { points: 0, max: 6, hint: 'Mangler revenue growth' }
  if (v > 0.15) return { points: 6, max: 6, hint: 'Revenue growth > 15% (+6)' }
  if (v >= 0.05) return { points: 3, max: 6, hint: 'Revenue growth mellem 5% og 15% (+3)' }
  if (v >= 0) return { points: 1, max: 6, hint: 'Revenue growth mellem 0% og 5% (+1)' }
  return { points: 0, max: 6, hint: 'Revenue shrinkage (+0)' }
}

function evalInsider(v?: number | null): Eval {
  if (v == null) return { points: 0, max: 6, hint: 'Mangler insider ejerskab' }
  if (v >= 0.1) return { points: 6, max: 6, hint: 'Insider ownership >= 10% (+6)' }
  if (v >= 0.03) return { points: 3, max: 6, hint: 'Insider ownership mellem 3% og 10% (+3)' }
  return { points: 0, max: 6, hint: 'Insider ownership < 3% (+0)' }
}

function evalCloseAbove(flag?: boolean | null): Eval {
  return flag ? { points: 8, max: 8, hint: 'Kursen ligger over SMA200 (+8)' } : { points: 0, max: 8, hint: 'Kursen ligger under SMA200 (+0)' }
}

function evalSmaTrend(flag?: boolean | null): Eval {
  return flag ? { points: 4, max: 4, hint: 'SMA50 ligger over SMA200 (+4)' } : { points: 0, max: 4, hint: 'SMA50 under SMA200 (+0)' }
}

function evalRsi(v?: number | null): Eval {
  if (v == null) return { points: 0, max: 10, hint: 'Mangler RSI data' }
  if (v >= 60 && v < 70) return { points: 10, max: 10, hint: 'RSI mellem 60 og 70 (+10)' }
  if ((v >= 45 && v < 60) || v > 80) return { points: 6, max: 10, hint: 'RSI mellem 45 og 60 eller > 80 (+6)' }
  return { points: 0, max: 10, hint: 'RSI uden momentum-bonus (+0)' }
}

function evalAtr(v?: number | null): Eval {
  if (v == null) return { points: 0, max: 6, hint: 'Mangler ATR data' }
  if (v >= 2 && v <= 6) return { points: 6, max: 6, hint: 'ATR% mellem 2% og 6% (+6)' }
  if (v >= 1 && v <= 8) return { points: 4, max: 6, hint: 'ATR% mellem 1% og 8% (+4)' }
  return { points: 1, max: 6, hint: 'ATR% udenfor sweet-spot (+1)' }
}

function evalVolume(rising?: boolean | null, priceAbove?: boolean | null): VolumeEval {
  if (rising && priceAbove) return { points: 7, max: 7, hint: 'Volumen stiger og prisen er over MA20 (+7)', value: 'Rising + price>MA20' }
  if (rising) return { points: 3, max: 7, hint: 'Volumen stiger (+3)', value: 'Rising' }
  return { points: 0, max: 7, hint: 'Volumen flad/falder (+0)', value: 'Flat' }
}

function evalSentimentMean(v?: number | null): Eval {
  if (v == null) return { points: 0, max: 15, hint: 'Ingen sentimentdata' }
  if (v > 0.2) return { points: 15, max: 15, hint: 'VADER > 0.2 (+15)' }
  if (v >= 0) return { points: 8, max: 15, hint: 'VADER mellem 0 og 0.2 (+8)' }
  return { points: 0, max: 15, hint: 'Negativ VADER (<0) (+0)' }
}

function evalFlow(value?: string | null): FlowEval {
  if (value === 'high') return { points: 6, max: 6, hint: 'News flow above 30-day baseline (+6)', value: 'High' }
  if (value === 'neutral') return { points: 3, max: 6, hint: 'News flow near baseline (+3)', value: 'Neutral' }
  return { points: 0, max: 6, hint: 'News flow below baseline (+0)', value: value ? value : 'Low' }
}

function evalSignal(flag?: boolean | null): SignalEval {
  return flag ? { points: 4, max: 4, hint: 'Strong signal terms found in headlines (+4)', value: 'Detected' } : { points: 0, max: 4, hint: 'Ingen signalord fundet (+0)', value: 'None' }
}

export default function TickerClient({ id }: { id: string }) {
  const [data, setData] = useState<StockData | null>(null)
  const [meta, setMeta] = useState<StockMeta | null>(null)
  const [range, setRange] = useState<'1M'|'3M'|'6M'|'1Y'>('3M')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [usingCache, setUsingCache] = useState(false)
  const [metaCached, setMetaCached] = useState(false)

  async function refreshMeta(force = false) {
    if (typeof window === 'undefined') return
    const cached = readCache<StockMeta>(META_CACHE_KEY)
    if (cached) {
      setMeta(cached.payload)
      setMetaCached(true)
      if (!force && isFresh(cached)) return
    }
    try {
      const json = await fetch(`${DATA_BASE}/meta.json`).then(r => { if (!r.ok) throw new Error(`meta ${r.status}`); return r.json() })
      setMeta(json)
      setMetaCached(false)
      writeCache(META_CACHE_KEY, json)
    } catch (e) {
      console.error('Failed to load meta', e)
    }
  }

  async function fetchTicker(opts?: { force?: boolean }) {
    if (typeof window === 'undefined') return
    const key = tickerCacheKey(id)
    const cached = readCache<StockData>(key)
    if (!opts?.force && cached && isFresh(cached)) {
      setData(cached.payload)
      setUsingCache(true)
      return
    }
    if (cached && !opts?.force) {
      setData(cached.payload)
      setUsingCache(true)
    }
    try {
      setLoading(true)
      setError(null)
      const json = await fetch(`${DATA_BASE}/${id.replace(/\s+/g,'_')}.json`).then(r => { if (!r.ok) throw new Error(`data ${r.status}`); return r.json() })
      setData(json)
      setUsingCache(false)
      writeCache(key, json)
    } catch (e) {
      console.error('Failed to load ticker', e)
      setError('Kunne ikke hente den seneste data for denne ticker.')
      if (cached) {
        setData(prev => prev ?? cached.payload)
        setUsingCache(true)
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (typeof window === 'undefined') return
    const cached = readCache<StockData>(tickerCacheKey(id))
    if (cached) {
      setData(cached.payload)
      setUsingCache(true)
    } else {
      setData(null)
      setUsingCache(false)
    }
    fetchTicker()
  }, [id])

  useEffect(() => {
    refreshMeta()
  }, [])

  function addToPortfolio(t: string, price?: number | null) {
    try {
      const raw = localStorage.getItem('portfolio')
      const arr: Array<{ticker:string; qty:number; avgCost:number}> = raw ? JSON.parse(raw) : []
      const qtyStr = window.prompt(`Quantity for ${t}?`, '1')
      if (qtyStr == null) return
      const qty = Number(qtyStr)
      const costDefault = price ?? 0
      const costStr = window.prompt(`Avg cost for ${t}?`, String(costDefault))
      if (costStr == null) return
      const avgCost = Number(costStr)
      const ix = arr.findIndex(x => x.ticker === t)
      const rec = { ticker: t, qty, avgCost }
      if (ix >= 0) arr[ix] = rec; else arr.push(rec)
      localStorage.setItem('portfolio', JSON.stringify(arr))
      alert('Saved to portfolio')
    } catch (e) {
      console.error('portfolio save failed', e)
    }
  }

  const scoreColor = (n?: number) => {
    const v = n ?? 0
    if (v >= 70) return 'green'
    if (v >= 50) return 'orange'
    return 'crimson'
  }
  const fmt = (n?: number | null) => (n == null ? '--' : new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(n))
  const fmtPercent = (ratio?: number | null) => (ratio == null ? '--' : `${(ratio * 100).toFixed(2)}%`)
  const fmtPercentDirect = (n?: number | null) => (n == null ? '--' : `${n.toFixed(2)}%`)

  const flags = useMemo(() => (data?.flags ?? []).filter(f => !String(f).includes('_fail')), [data])
  const F: any = (data as any)?.fundamentals || {}
  const T: any = (data as any)?.technicals || {}
  const S: any = (data as any)?.sentiment || {}

  function rangeToWindow(r: '1M'|'3M'|'6M'|'1Y') { return r==='1M'?22 : r==='3M'?66 : r==='6M'?120 : 252 }
  const series: any = T?.series || {}
  const win = rangeToWindow(range)
  const slicer = (arr?: any[]) => (Array.isArray(arr) ? arr.slice(-win) : [])
  const labels: string[] = slicer(series.dates)
  const priceClose: Array<number|null> = slicer(series.close)
  const priceSMA50: Array<number|null> = slicer(series.sma50)
  const priceSMA200: Array<number|null> = slicer(series.sma200)
  const rsiArr: Array<number|null> = slicer(series.rsi)
  const atrArr: Array<number|null> = slicer(series.atr_pct)

  const fundamentalRows = useMemo(() => ([
    { label: 'FCF Yield', value: fmtPercent(F?.fcf_yield), ...evalFcfYield(F?.fcf_yield) },
    { label: 'NetDebt/EBITDA', value: fmt(F?.nd_to_ebitda), ...evalNetDebt(F?.nd_to_ebitda) },
    { label: 'Gross Margin', value: fmtPercent(F?.gross_margin), ...evalGrossMargin(F?.gross_margin) },
    { label: 'Revenue Growth', value: fmtPercent(F?.revenue_growth), ...evalRevenueGrowth(F?.revenue_growth) },
    { label: 'Insider Own', value: fmtPercent(F?.insider_own), ...evalInsider(F?.insider_own) },
  ]), [F])

  const trendBadges = useMemo(() => ([
    { label: 'Close > SMA200', ok: T?.trend?.close_gt_sma200 === true, ...evalCloseAbove(T?.trend?.close_gt_sma200 === true) },
    { label: 'SMA50 > SMA200', ok: T?.trend?.sma50_gt_sma200 === true, ...evalSmaTrend(T?.trend?.sma50_gt_sma200 === true) },
    { label: 'Price > MA20', ok: T?.vol_confirm?.price_gt_ma20 === true, points: 0, max: 0, hint: 'Bruges sammen med volumen til at give op til 7 point' },
  ]), [T])

  const rsiEval = useMemo(() => evalRsi(T?.rsi), [T])
  const atrEval = useMemo(() => evalAtr(T?.atr_pct), [T])
  const volumeEval = useMemo(() => evalVolume(T?.vol_confirm?.rising20, T?.vol_confirm?.price_gt_ma20), [T])

  const sentimentMeanEval = useMemo(() => evalSentimentMean(S?.mean7), [S])
  const flowEval = useMemo(() => evalFlow(S?.flow), [S])
  const signalEval = useMemo(() => evalSignal(S?.signal_terms), [S])

  const newsUrl = useMemo(() => googleNewsUrl(id), [id])

  return (
    <section style={{display:'grid', gap:16}}>{/* Container */}
      <div style={{display:'flex', alignItems:'center', justifyContent:'space-between', gap:12, flexWrap:'wrap'}}>{/* Header */}
        <div style={{display:'flex', alignItems:'center', gap:12, flexWrap:'wrap'}}>{/* Title block */}
          <h1 style={{margin:'8px 0'}}>{id}</h1>
          <span style={{padding:'4px 10px', borderRadius:16, background:'#f4f7ff', border:'1px solid #e3e9ff', color:scoreColor(data?.score), fontWeight:600}}>Score: {data?.score ?? '--'}</span>
          <small style={{color:'#666'}}>Updated: {data?.updated_at ?? '--'}{usingCache ? ' (cache)' : ''}</small>
          <small style={{color:'#666'}}>Data refresh: {timeAgo(meta?.generated_at)}{metaCached ? ' (cache)' : ''}</small>
        </div>
        <div style={{display:'flex', gap:8}}>{/* Actions */}
          <a href={`${BASE}/`} style={{textDecoration:'none'}}>&larr; Back</a>
          <button onClick={() => { refreshMeta(true); fetchTicker({ force: true }) }} disabled={loading} style={btn}>{loading ? 'Refreshing...' : 'Refresh'}</button>
          <button onClick={() => addToPortfolio(id, (data as any)?.price)} style={btn}>Add to portfolio</button>
        </div>
      </div>

      {error && !data ? (
        <div style={card}>
          <h3 style={h3}>Ingen data</h3>
          <p style={pMuted}>Could not load data for this ticker. Try refreshing or come back later.</p>
        </div>
      ) : null}

      {error && data ? (
        <div style={alertWarn}>Could not refresh data. Showing the latest cached value.</div>
      ) : null}

      {flags.length > 0 && (
        <div style={{display:'flex', gap:8, flexWrap:'wrap'}}>{flags.map((f, i) => (
          <span key={i} style={chipWarn}>{f}</span>
        ))}</div>
      )}

      <div style={grid2}>{/* Price & trend */}
        <div style={card}>
          <h3 style={h3}>Price</h3>
          <div style={{display:'flex', gap:24, flexWrap:'wrap'}}>{/* Price metrics */}
            <Metric label="Last" value={fmt((data as any)?.price)} />
            <Metric label="SMA50" value={fmt((data as any)?.sma50)} />
            <Metric label="SMA200" value={fmt((data as any)?.sma200)} />
          </div>
          <div style={{display:'flex', gap:8, alignItems:'center', flexWrap:'wrap', margin:'4px 0 8px'}}>{/* Range picker */}
            <small style={{color:'#666'}}>Range:</small>
            {(['1M','3M','6M','1Y'] as const).map(r => (
              <button key={r} onClick={()=>setRange(r)} style={{...btnMini, background: range===r ? '#eef' : '#fff'}}>{r}</button>
            ))}
          </div>
          {(data?.price == null || flags.some(f=>String(f).includes('no_price_data'))) && (
            <span style={chipWarn} title="Yahoo price missing for this ticker">Missing data</span>
          )}
          {labels.length ? (
            <div style={{height:160, marginTop:8}}>{/* Price chart */}
              <PriceChart labels={labels} close={priceClose} sma50={priceSMA50} sma200={priceSMA200} />
            </div>
          ) : null}
        </div>
        <div style={card}>
          <h3 style={h3}>Trend <span style={scoreTag}>{data?.tech_points ?? 0}/35</span></h3>
          <div style={{display:'flex', gap:16, flexWrap:'wrap'}}>{trendBadges.map((b, i) => (
            <Badge key={i} ok={b.ok} label={b.label} points={b.points} max={b.max} hint={b.hint} />
          ))}</div>
        </div>
      </div>

      <div style={grid3}>{/* Metrics cards */}
        <div style={card}>
          <h3 style={h3}>Buy Box</h3>
          <div style={{display:'flex', gap:8, flexWrap:'wrap'}}>
            {(((data as any)?.buy_zones as any[]) || []).map((z: any, i: number) => (
              <span key={i} style={chip} title={z?.rationale || ''}>
                {z?.type === 'sma_pullback' ? `${z?.ma ?? 'SMA'}: ${fmt(z?.price_low)}–${fmt(z?.price_high)}` :
                 z?.type === 'breakout_retest' ? `Retest: ${fmt(z?.price_low)}–${fmt(z?.price_high)}` :
                 `${z?.type ?? 'zone'}: ${fmt(z?.price_low)}–${fmt(z?.price_high)}`}
              </span>
            ))}
            {(!(((data as any)?.buy_zones as any[]) || []).length) ? <small style={{color:'#666'}}>Ingen købszoner endnu</small> : null}
          </div>
        </div>

        <div style={card}>
          <h3 style={h3}>Fundamentals <span style={scoreTag}>{data?.fund_points ?? 0}/40</span></h3>
          <div style={grid2min}>{fundamentalRows.map(row => (
            <Metric key={row.label} label={row.label} value={row.value} points={row.points} max={row.max} hint={row.hint} />
          ))}</div>
        </div>
        <div style={card}>
          <h3 style={h3}>Technicals <span style={scoreTag}>{data?.tech_points ?? 0}/35</span></h3>
          <div style={grid2min}>
            <Metric label="RSI(14)" value={fmt(T?.rsi)} accent={rsiColor(T?.rsi)} points={rsiEval.points} max={rsiEval.max} hint={rsiEval.hint} />
            <Metric label="ATR%" value={fmtPercentDirect(T?.atr_pct)} points={atrEval.points} max={atrEval.max} hint={atrEval.hint} />
            <Metric label="Volume confirm" value={volumeEval.value} points={volumeEval.points} max={volumeEval.max} hint={volumeEval.hint} />
          </div>
          {rsiArr.length ? (
            <div style={{height:120, marginTop:8}}>{/* RSI chart */}
              <LineChart labels={labels} data={rsiArr} label="RSI(14)" color="#805ad5" ySuggestedMin={0} ySuggestedMax={100} />
            </div>
          ) : null}
          {atrArr.length ? (
            <div style={{height:120, marginTop:8}}>{/* ATR chart */}
              <LineChart labels={labels} data={atrArr} label="ATR%" color="#3182ce" />
            </div>
          ) : null}
        </div>
        <div style={card}>
          <h3 style={h3}>Sentiment <span style={scoreTag}>{data?.sent_points ?? 0}/25</span></h3>
          <div style={grid2min}>
            <Metric label="Mean(7d)" value={fmt(S?.mean7)} points={sentimentMeanEval.points} max={sentimentMeanEval.max} hint={sentimentMeanEval.hint} />
            <Metric label="# News 7d" value={S?.count7 ?? '--'} hint="Antal artikler de seneste 7 dage" />
            <Metric label="# News 30d" value={S?.count30 ?? '--'} hint="Antal artikler de seneste 30 dage" />
            <Metric label="Flow" value={flowEval.value} points={flowEval.points} max={flowEval.max} hint={flowEval.hint} />
            <Metric label="Signal terms" value={signalEval.value} points={signalEval.points} max={signalEval.max} hint={signalEval.hint} />
          </div>
          <div style={{marginTop:8}}>{/* Sentiment actions */}
            <a href={newsUrl} target="_blank" rel="noopener noreferrer" style={btn} title="Open Google News">View on Google News -&gt;</a>
          </div>
        </div>
      </div>
    </section>
  )
}

function Metric({ label, value, accent, points, max, hint }: { label: string; value: any; accent?: string; points?: number; max?: number; hint?: string }) {
  const hasPoints = typeof points === 'number' && typeof max === 'number' && max > 0
  const title = hint ? `${hint}${hasPoints ? ` (${points}/${max} pts)` : ''}` : hasPoints ? `${points}/${max} pts` : undefined
  const display = value == null || value === '' ? '--' : value
  return (
    <div style={{minWidth:140}} title={title}>
      <div style={{fontSize:12, color:'#666'}}>{label}</div>
      <div style={{fontSize:18, fontWeight:600, color: accent ?? '#222'}}>{display}</div>
      {hasPoints ? <div style={metricPoints}>{points}/{max} pts</div> : null}
    </div>
  )
}

function Badge({ ok, label, points, max, hint }: { ok: boolean; label: string; points?: number; max?: number; hint?: string }) {
  const text = points != null && max ? `${label} (${points}/${max})` : label
  const title = hint ? `${hint}${points != null && max ? ` (${points}/${max} pts)` : ''}` : undefined
  return (
    <span style={{...chip, background: ok ? '#eef9f0' : '#fff5f5', borderColor: ok ? '#c8efd2' : '#ffd6d6', color: ok ? '#216e39' : '#a94442'}} title={title}>
      {ok ? 'OK' : 'X'} {text}
    </span>
  )
}

function rsiColor(rsi?: number | null) {
  if (rsi == null) return '#222'
  if (rsi >= 70) return '#a94442'
  if (rsi <= 30) return '#216e39'
  return '#222'
}

function timeAgo(iso?: string) {
  if (!iso) return '--'
  const d = new Date(iso)
  if (isNaN(d.getTime())) return '--'
  const diffMs = Date.now() - d.getTime()
  const mins = Math.floor(diffMs / 60000)
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 48) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}


const btn: React.CSSProperties = { padding:'6px 10px', border:'1px solid #ddd', background:'#fafafa', cursor:'pointer' }
const btnMini: React.CSSProperties = { padding:'4px 8px', border:'1px solid #ddd', background:'#fff', cursor:'pointer', borderRadius:6 }

const card: React.CSSProperties = { padding:12, border:'1px solid #eee', borderRadius:8, background:'#fff' }
const h3: React.CSSProperties = { margin:'0 0 8px 0', fontSize:14, color:'#444', display:'flex', alignItems:'center', gap:8 }
const chip: React.CSSProperties = { display:'inline-block', padding:'4px 8px', border:'1px solid #eee', borderRadius:16, fontSize:12 }
const chipWarn: React.CSSProperties = { ...chip, background:'#fff8e6', borderColor:'#ffe6b3', color:'#8a6d3b' }
const grid2: React.CSSProperties = { display:'grid', gap:12, gridTemplateColumns:'repeat(auto-fit, minmax(260px, 1fr))' }
const grid3: React.CSSProperties = { display:'grid', gap:12, gridTemplateColumns:'repeat(auto-fit, minmax(240px, 1fr))' }
const grid2min: React.CSSProperties = { display:'grid', gap:8, gridTemplateColumns:'repeat(auto-fit, minmax(140px, 1fr))' }
const alertWarn: React.CSSProperties = { padding:'8px 12px', background:'#fff4e5', border:'1px solid #ffd8a8', borderRadius:8, color:'#8a6d3b' }
const scoreTag: React.CSSProperties = { fontSize:12, fontWeight:500, color:'#555', background:'#f2f2f2', padding:'2px 8px', borderRadius:12 }
const metricPoints: React.CSSProperties = { fontSize:12, color:'#667' }
const pMuted: React.CSSProperties = { margin:'4px 0 0', color:'#555' }
