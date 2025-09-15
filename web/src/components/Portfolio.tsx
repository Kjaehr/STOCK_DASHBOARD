"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockMeta, StockData } from '../types'

type Holding = { ticker: string; qty: number; avgCost: number }

const LS_KEY = 'portfolio'

export default function Portfolio() {
  const [holdings, setHoldings] = useState<Holding[]>([])
  const [meta, setMeta] = useState<StockMeta | null>(null)
  const [dataMap, setDataMap] = useState<Record<string, StockData>>({})

  // load holdings
  useEffect(() => {
    try {
      const raw = localStorage.getItem(LS_KEY)
      if (raw) setHoldings(JSON.parse(raw))
    } catch {}
  }, [])

  // persist holdings
  useEffect(() => {
    try { localStorage.setItem(LS_KEY, JSON.stringify(holdings)) } catch {}
  }, [holdings])

  // load latest prices for tickers present in holdings
  useEffect(() => {
    const run = async () => {
      try {
        const m = await fetch('/data/meta.json').then(r => r.json()) as StockMeta
        setMeta(m)
        const wanted = Array.from(new Set(holdings.map(h => h.ticker)))
        const results = await Promise.allSettled(wanted.map(t => fetch(`/data/${t.replace(/\s+/g,'_')}.json`).then(r => r.json())))
        const ok = results.flatMap(r => r.status === 'fulfilled' ? [r.value as StockData] : [])
        const map: Record<string, StockData> = {}
        for (const s of ok) map[s.ticker] = s
        setDataMap(map)
      } catch (e) {
        console.error('Failed to load data', e)
      }
    }
    if (holdings.length) run()
  }, [holdings])

  const [form, setForm] = useState<Holding>({ ticker: '', qty: 0, avgCost: 0 })

  const rows = useMemo(() => {
    return holdings.map(h => {
      const s = dataMap[h.ticker]
      const price = s?.price ?? null
      const value = price != null ? price * h.qty : null
      const gain = price != null ? (price - h.avgCost) * h.qty : null
      const gainPct = price != null && h.avgCost ? ((price - h.avgCost) / h.avgCost) * 100 : null
      return { ...h, price, value, gain, gainPct, score: s?.score ?? null }
    })
  }, [holdings, dataMap])

  const totals = useMemo(() => {
    const totalValue = rows.reduce((a, r) => a + (r.value ?? 0), 0)
    return { totalValue }
  }, [rows])

  function addHolding() {
    const t = form.ticker.trim().toUpperCase()
    if (!t) return
    setHoldings(prev => {
      const next = [...prev]
      const ix = next.findIndex(x => x.ticker === t)
      if (ix >= 0) next[ix] = { ticker: t, qty: form.qty, avgCost: form.avgCost }
      else next.push({ ticker: t, qty: form.qty, avgCost: form.avgCost })
      return next
    })
    setForm({ ticker: '', qty: 0, avgCost: 0 })
  }

  function removeHolding(t: string) {
    setHoldings(prev => prev.filter(x => x.ticker !== t))
  }

  function onImport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const fr = new FileReader()
    fr.onload = () => {
      try {
        const arr = JSON.parse(String(fr.result)) as Holding[]
        if (Array.isArray(arr)) setHoldings(arr)
      } catch {}
    }
    fr.readAsText(file)
  }

  function onExport() {
    const blob = new Blob([JSON.stringify(holdings, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'portfolio.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section>
      <h1 style={{margin:'8px 0'}}>Portfolio</h1>
      <div style={{display:'flex', gap:12, flexWrap:'wrap', alignItems:'end'}}>
        <div>
          <label>Ticker<br/>
            <input value={form.ticker} onChange={e=>setForm({...form, ticker:e.target.value})} placeholder="e.g., APLD" style={inp} />
          </label>
        </div>
        <div>
          <label>Qty<br/>
            <input type="number" value={form.qty} onChange={e=>setForm({...form, qty:Number(e.target.value)})} style={inp} />
          </label>
        </div>
        <div>
          <label>Avg cost<br/>
            <input type="number" value={form.avgCost} onChange={e=>setForm({...form, avgCost:Number(e.target.value)})} style={inp} />
          </label>
        </div>
        <button onClick={addHolding} style={btn}>Add/Update</button>
        <input type="file" accept="application/json" onChange={onImport} />
        <button onClick={onExport} style={btn}>Export JSON</button>
      </div>

      <div style={{marginTop:12}}>
        <strong>Total value:</strong> {fmt(totals.totalValue)}
      </div>

      <table style={{width:'100%', borderCollapse:'collapse', marginTop:12}}>
        <thead>
          <tr>
            <th style={th}>Ticker</th>
            <th style={th}>Qty</th>
            <th style={th}>Avg cost</th>
            <th style={th}>Price</th>
            <th style={th}>Value</th>
            <th style={th}>Gain</th>
            <th style={th}>Gain %</th>
            <th style={th}>Score</th>
            <th style={th}></th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.ticker}>
              <td style={td}>{r.ticker}</td>
              <td style={td}>{num(r.qty)}</td>
              <td style={td}>{fmt(r.avgCost)}</td>
              <td style={td}>{fmt(r.price)}</td>
              <td style={td}>{fmt(r.value)}</td>
              <td style={{...td, color: (r.gain ?? 0) >= 0 ? 'green' : 'crimson'}}>{fmt(r.gain)}</td>
              <td style={td}>{pct(r.gainPct)}</td>
              <td style={{...td, color: scoreColor(r.score ?? 0)}}>{r.score ?? '—'}</td>
              <td style={td}><button onClick={()=>removeHolding(r.ticker)} style={btn}>Remove</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

const inp: React.CSSProperties = { padding:'8px', border:'1px solid #ddd', minWidth:120 }
const btn: React.CSSProperties = { padding:'8px 10px', border:'1px solid #ddd', background:'#fafafa', cursor:'pointer' }
const th: React.CSSProperties = { textAlign:'left', borderBottom:'1px solid #eee', padding:'8px' }
const td: React.CSSProperties = { borderBottom:'1px solid #f2f2f2', padding:'8px' }

function fmt(n?: number | null) {
  if (n == null) return '—'
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(n)
}
function num(n?: number | null) {
  if (n == null) return '—'
  return new Intl.NumberFormat().format(n)
}
function pct(n?: number | null) {
  if (n == null) return '—'
  return `${n.toFixed(2)}%`
}
function scoreColor(v: number) {
  if (v >= 70) return 'green'
  if (v >= 50) return 'orange'
  return 'crimson'
}

