"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockMeta, StockData } from '../types'
import { BASE } from '../base'

export default function Leaderboard() {
  const [meta, setMeta] = useState<StockMeta | null>(null)
  const [items, setItems] = useState<StockData[]>([])
  const [q, setQ] = useState('')
  const [loading, setLoading] = useState(false)
  const [preset, setPreset] = useState<'NONE'|'HIGH_MOM'|'UNDERVALUED'|'LOW_ATR'>('NONE')
  const [onlyPriced, setOnlyPriced] = useState(false)


  async function fetchAll() {
    try {
      setLoading(true)
      const m = await fetch(`${BASE}/data/meta.json`).then(r => r.json()) as StockMeta
      setMeta(m)
      const list = (m.tickers ?? [])
      const results = await Promise.allSettled(
        list.map(t => fetch(`${BASE}/data/${t.replace(/\s+/g,'_')}.json`).then(r => r.json()))
      )
      const ok = results.flatMap(r => r.status === 'fulfilled' ? [r.value as StockData] : [])
      setItems(ok)
    } catch (e) {
      console.error('Failed to load data', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchAll() }, [])

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase()
    let arr = items
    if (s) arr = arr.filter(x => x.ticker?.toLowerCase().includes(s))
    if (onlyPriced) arr = arr.filter(x => x.price != null && !(x.flags||[]).some(f => String(f).includes('no_price_data')))
    if (preset === 'HIGH_MOM') {
      arr = arr.filter(x => {
        const rsi = (x as any)?.technicals?.rsi
        return typeof rsi === 'number' && rsi > 60
      })
    } else if (preset === 'UNDERVALUED') {
      arr = arr.filter(x => {
        const fy = (x as any)?.fundamentals?.fcf_yield
        return typeof fy === 'number' && fy >= 8
      })
    } else if (preset === 'LOW_ATR') {
      arr = arr.filter(x => {
        const atr = (x as any)?.technicals?.atr_pct
        return typeof atr === 'number' && atr > 0 && atr < 8
      })
    }
    return arr.sort((a,b) => (b.score ?? 0) - (a.score ?? 0))
  }, [items, q, preset, onlyPriced])

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

  return (
    <section>
      <h1 style={{margin:'8px 0'}}>Leaderboard</h1>
      <div style={{display:'flex', gap:12, alignItems:'center'}}>
        <input placeholder="Search ticker" value={q} onChange={e=>setQ(e.target.value)} style={{padding:'8px', border:'1px solid #ddd'}} />
        <button onClick={fetchAll} disabled={loading} style={btnMini}>{loading ? 'Refreshing…' : 'Refresh'}</button>
        <small style={{color:'#666'}}>Updated: {meta?.generated_at ?? '—'}</small>
        <select value={preset} onChange={e=>setPreset(e.target.value as any)} style={{padding:'6px', border:'1px solid #ddd'}}>
          <option value="NONE">Presets</option>
          <option value="HIGH_MOM">High momentum</option>
          <option value="UNDERVALUED">Undervalued</option>
          <option value="LOW_ATR">Low ATR%</option>
        </select>
        <label style={{display:'flex', alignItems:'center', gap:6}}>
          <input type="checkbox" checked={onlyPriced} onChange={e=>setOnlyPriced(e.target.checked)} />
          Only with price
        </label>
      </div>
      <table style={{width:'100%', borderCollapse:'collapse', marginTop:12}}>
        <thead>
          <tr>
            <th style={th}>Ticker</th>
            <th style={th}>Score</th>
            <th style={th}>Price</th>
            <th style={th}>Fund</th>
            <th style={th}>Tech</th>
            <th style={th}>Sent</th>
            <th style={th}>Flags</th>
        <select value={preset} onChange={e=>setPreset(e.target.value as any)} style={{padding:'6px', border:'1px solid #ddd'}}>
          <option value="NONE">Presets</option>
          <option value="HIGH_MOM">High momentum</option>
          <option value="UNDERVALUED">Undervalued</option>
          <option value="LOW_ATR">Low ATR%</option>
        </select>
        <label style={{display:'flex', alignItems:'center', gap:6}}>
          <input type="checkbox" checked={onlyPriced} onChange={e=>setOnlyPriced(e.target.checked)} />
          Only with price
        </label>
            <th style={th}></th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(row => (
            <tr key={row.ticker}>
              <td style={td}>{row.ticker}</td>
              <td style={{...td, color: scoreColor(row.score)}}>{row.score ?? 0}</td>
              <td style={td}>
                {fmt(row.price)}{' '}
                {(row.price == null || (row.flags||[]).some(f => f.includes('no_price_data')))
                  ? <span style={badgeWarn} title="Yahoo price missing for this ticker">Missing data</span>
                  : null}
              </td>
              <td style={td}>{row.fund_points ?? 0}</td>
              <td style={td}>{row.tech_points ?? 0}</td>
              <td style={td}>{row.sent_points ?? 0}</td>
              <td style={td}><small>{(row.flags||[]).join(', ')}</small></td>
              <td style={td}>
                <a href={`/ticker/${encodeURIComponent(row.ticker)}`}>Details</a>
                {' '}
                <button onClick={()=>addToPortfolio(row.ticker, row.price)} style={btnMini}>Add</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

const th: React.CSSProperties = { textAlign:'left', borderBottom:'1px solid #eee', padding:'8px' }
const td: React.CSSProperties = { borderBottom:'1px solid #f2f2f2', padding:'8px' }
const btnMini: React.CSSProperties = { padding:'4px 8px', border:'1px solid #ddd', background:'#fafafa', cursor:'pointer', marginLeft:8 }
const badgeWarn: React.CSSProperties = { display:'inline-block', padding:'2px 6px', borderRadius:12, background:'#fff3cd', color:'#8a6d3b', border:'1px solid #ffe69c', fontSize:12 }


function scoreColor(n?: number) {
  const v = n ?? 0
  if (v >= 70) return 'green'
  if (v >= 50) return 'orange'
  return 'crimson'
}

function fmt(n?: number | null) {
  if (n == null) return '—'
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(n)
}

