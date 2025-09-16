"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockData, StockMeta } from '../../../types'
import { PriceChart, LineChart } from '../../../components/Charts'
import { BASE } from '../../../base'

export default function TickerClient({ id }: { id: string }) {
  const [data, setData] = useState<StockData | null>(null)
  const [meta, setMeta] = useState<StockMeta | null>(null)
  const [range, setRange] = useState<'1M'|'3M'|'6M'|'1Y'>('3M')
  const [loading, setLoading] = useState(false)

  async function fetchOne() {
    try {
      setLoading(true)
      const json = await fetch(`${BASE}/data/${id.replace(/\s+/g,'_')}.json`).then(r => r.json())
      setData(json)
    } catch (e) {
      console.error('Failed to load ticker', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchOne() }, [id])

  useEffect(() => {
    (async () => {
      try { const m = await fetch(`${BASE}/data/meta.json`).then(r=>r.json()); setMeta(m) } catch {}
    })()
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
  const fmt = (n?: number | null) => (n == null ? '—' : new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(n))
  const pct = (n?: number | null) => (n == null ? '—' : `${(n).toFixed(2)}%`)

  const flags = useMemo(() => (data?.flags ?? []), [data])
  const F: any = (data as any)?.fundamentals || {}
  const T: any = (data as any)?.technicals || {}

  // Chart range helpers and derived series
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

  const S: any = (data as any)?.sentiment || {}

  return (
    <section style={{display:'grid', gap:16}}>
      <div style={{display:'flex', alignItems:'center', justifyContent:'space-between', gap:12, flexWrap:'wrap'}}>
        <div style={{display:'flex', alignItems:'center', gap:12, flexWrap:'wrap'}}>
          <h1 style={{margin:'8px 0'}}>{id}</h1>
          <span style={{padding:'4px 10px', borderRadius:16, background:'#f4f7ff', border:'1px solid #e3e9ff', color:scoreColor(data?.score), fontWeight:600}}>Score: {data?.score ?? '—'}</span>
          <small style={{color:'#666'}}>Updated: {data?.updated_at ?? '—'}</small>
          <small style={{color:'#666'}}>Data refresh: {timeAgo(meta?.generated_at)}</small>
        </div>
        <div style={{display:'flex', gap:8}}>
          <a href="/" style={{textDecoration:'none'}}>&larr; Back</a>
          <button onClick={fetchOne} disabled={loading} style={btn}> {loading ? 'Refreshing…' : 'Refresh'} </button>

          <button onClick={()=>addToPortfolio(id, (data as any)?.price)} style={btn}>Add to portfolio</button>
        </div>
      </div>

      {flags.length > 0 && (
        <div style={{display:'flex', gap:8, flexWrap:'wrap'}}>
          {flags.map((f, i) => (
            <span key={i} style={chipWarn}>{f}</span>
          ))}
        </div>
      )}

      <div style={grid2}>
        <div style={card}>
          <h3 style={h3}>Price</h3>
          <div style={{display:'flex', gap:24, flexWrap:'wrap'}}>
            <Metric label="Last" value={fmt(data?.price)} />
            <Metric label="SMA50" value={fmt((data as any)?.sma50)} />
            <Metric label="SMA200" value={fmt((data as any)?.sma200)} />
          </div>
          <div style={{display:'flex', gap:8, alignItems:'center', flexWrap:'wrap', margin:'4px 0 8px'}}>
            <small style={{color:'#666'}}>Range:</small>
            {(['1M','3M','6M','1Y'] as const).map(r => (
              <button key={r} onClick={()=>setRange(r)} style={{...btnMini, background: range===r ? '#eef' : '#fff'}}>{r}</button>
            ))}
          </div>
          {(data?.price == null || flags.some(f=>String(f).includes('no_price_data'))) && (
            <span style={chipWarn} title="Yahoo price missing for this ticker">Missing data</span>
          )}
          {labels.length ? (
            <div style={{height:160, marginTop:8}}>
              <PriceChart labels={labels} close={priceClose} sma50={priceSMA50} sma200={priceSMA200} />
            </div>
          ) : null}
        </div>
        <div style={card}>
          <h3 style={h3}>Trend</h3>
          <div style={{display:'flex', gap:16, flexWrap:'wrap'}}>
            <Badge ok={(T?.trend?.close_gt_sma200) === true} label="Close > SMA200" />
            <Badge ok={(T?.trend?.sma50_gt_sma200) === true} label="SMA50 > SMA200" />
            <Badge ok={(T?.vol_confirm?.price_gt_ma20) === true} label="Price > MA20" />
          </div>
        </div>
      </div>

      <div style={grid3}>
        <div style={card}>
          <h3 style={h3}>Fundamentals</h3>
          <div style={grid2min}>
            <Metric label="FCF Yield" value={pct(F?.fcf_yield)} />
            <Metric label="NetDebt/EBITDA" value={F?.nd_to_ebitda == null ? '—' : fmt(F?.nd_to_ebitda)} />
            <Metric label="Gross Margin" value={pct(F?.gross_margin ? F?.gross_margin * 100 : null)} />
            <Metric label="Rev Growth" value={pct(F?.revenue_growth ? F?.revenue_growth * 100 : null)} />
            <Metric label="Insider Own" value={pct(F?.insider_own ? F?.insider_own * 100 : null)} />
          </div>
        </div>
        <div style={card}>
          <h3 style={h3}>Technicals</h3>
          <div style={grid2min}>
            <Metric label="RSI(14)" value={fmt(T?.rsi)} accent={rsiColor(T?.rsi)} />
            <Metric label="ATR%" value={pct(T?.atr_pct)} />
            <Metric label="Rising Vol(20)" value={(T?.vol_confirm?.rising20) ? 'Yes' : 'No'} />
          </div>
          {rsiArr.length ? (
            <div style={{height:120, marginTop:8}}>
              <LineChart labels={labels} data={rsiArr} label="RSI(14)" color="#805ad5" ySuggestedMin={0} ySuggestedMax={100} />
            </div>
          ) : null}
          {atrArr.length ? (
            <div style={{height:120, marginTop:8}}>
              <LineChart labels={labels} data={atrArr} label="ATR%" color="#3182ce" />
            </div>
          ) : null}
        </div>
        <div style={card}>
          <h3 style={h3}>Sentiment</h3>
          <div style={grid2min}>
            <Metric label="Mean(7d)" value={fmt(S?.mean7)} />
            <Metric label="# News 7d" value={S?.count7 ?? '—'} />
            <Metric label="# News 30d" value={S?.count30 ?? '—'} />
            <Metric label="Flow" value={S?.flow ?? '—'} />
          </div>
        </div>
      </div>
    </section>
  )
}

function Metric({ label, value, accent }: { label: string; value: any; accent?: string }) {
  return (
    <div style={{minWidth:120}}>
      <div style={{fontSize:12, color:'#666'}}>{label}</div>
      <div style={{fontSize:18, fontWeight:600, color: accent ?? '#222'}}>{value}</div>
    </div>
  )
}

function Badge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span style={{...chip, background: ok ? '#eef9f0' : '#fff5f5', borderColor: ok ? '#c8efd2' : '#ffd6d6', color: ok ? '#216e39' : '#a94442'}}>
      {ok ? '✓' : '✗'} {label}
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
  if (!iso) return '—'
  const d = new Date(iso)
  if (isNaN(d.getTime())) return '—'
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
const h3: React.CSSProperties = { margin:'0 0 8px 0', fontSize:14, color:'#444' }
const chip: React.CSSProperties = { display:'inline-block', padding:'4px 8px', border:'1px solid #eee', borderRadius:16, fontSize:12 }
const chipWarn: React.CSSProperties = { ...chip, background:'#fff8e6', borderColor:'#ffe6b3', color:'#8a6d3b' }
const grid2: React.CSSProperties = { display:'grid', gap:12, gridTemplateColumns:'repeat(auto-fit, minmax(260px, 1fr))' }
const grid3: React.CSSProperties = { display:'grid', gap:12, gridTemplateColumns:'repeat(auto-fit, minmax(240px, 1fr))' }
const grid2min: React.CSSProperties = { display:'grid', gap:8, gridTemplateColumns:'repeat(auto-fit, minmax(140px, 1fr))' }

