"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockMeta, StockData } from '../types'

export default function Leaderboard() {
  const [meta, setMeta] = useState<StockMeta | null>(null)
  const [items, setItems] = useState<StockData[]>([])
  const [q, setQ] = useState('')

  useEffect(() => {
    const run = async () => {
      try {
        const m = await fetch('/data/meta.json').then(r => r.json()) as StockMeta
        setMeta(m)
        const list = m.tickers ?? []
        const results = await Promise.allSettled(list.map(t => fetch(`/data/${t.replace(/\s+/g,'_')}.json`).then(r => r.json())))
        const ok = results.flatMap(r => r.status === 'fulfilled' ? [r.value as StockData] : [])
        setItems(ok)
      } catch (e) {
        console.error('Failed to load data', e)
      }
    }
    run()
  }, [])

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase()
    let arr = items
    if (s) arr = items.filter(x => x.ticker?.toLowerCase().includes(s))
    return arr.sort((a,b) => (b.score ?? 0) - (a.score ?? 0))
  }, [items, q])

  return (
    <section>
      <h1 style={{margin:'8px 0'}}>Leaderboard</h1>
      <div style={{display:'flex', gap:12, alignItems:'center'}}>
        <input placeholder="Search ticker" value={q} onChange={e=>setQ(e.target.value)} style={{padding:'8px', border:'1px solid #ddd'}} />
        <small style={{color:'#666'}}>Updated: {meta?.generated_at ?? '—'}</small>
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
            <th style={th}></th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(row => (
            <tr key={row.ticker}>
              <td style={td}>{row.ticker}</td>
              <td style={{...td, color: scoreColor(row.score)}}>{row.score ?? 0}</td>
              <td style={td}>{fmt(row.price)}</td>
              <td style={td}>{row.fund_points ?? 0}</td>
              <td style={td}>{row.tech_points ?? 0}</td>
              <td style={td}>{row.sent_points ?? 0}</td>
              <td style={td}><small>{(row.flags||[]).join(', ')}</small></td>
              <td style={td}><a href={`/ticker/${encodeURIComponent(row.ticker)}`}>Details</a></td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

const th: React.CSSProperties = { textAlign:'left', borderBottom:'1px solid #eee', padding:'8px' }
const td: React.CSSProperties = { borderBottom:'1px solid #f2f2f2', padding:'8px' }

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

