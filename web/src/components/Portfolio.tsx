"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockMeta, StockData } from '../types'
import { BASE } from '../base'

type Holding = { ticker: string; qty: number; avgCost: number }

const LS_KEY = 'portfolio'
const FILE_ACCEPT = '.json,.csv,application/json,text/csv'

function normalizeHolding(input: Partial<Holding>): Holding | null {
  if (!input.ticker) return null
  const ticker = String(input.ticker).trim().toUpperCase()
  if (!ticker) return null
  const qty = Number((input as any).qty)
  const avgCost = Number((input as any).avgCost)
  if (!Number.isFinite(qty) || !Number.isFinite(avgCost)) return null
  return { ticker, qty, avgCost }
}

function parseCsv(text: string): Holding[] {
  const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean)
  if (!lines.length) return []
  let startIndex = 0
  let tickerIndex = 0
  let qtyIndex = 1
  let costIndex = 2
  const first = lines[0].split(',').map(v => v.trim().toLowerCase())
  if (first.some(cell => cell.includes('ticker') || cell.includes('symbol'))) {
    const find = (...keys: string[]) => first.findIndex(cell => keys.includes(cell))
    const ti = find('ticker', 'symbol')
    const qi = find('qty', 'quantity', 'shares')
    const ci = find('avg_cost', 'avgcost', 'avg price', 'average_cost', 'averagecost', 'price')
    tickerIndex = ti >= 0 ? ti : tickerIndex
    qtyIndex = qi >= 0 ? qi : qtyIndex
    costIndex = ci >= 0 ? ci : costIndex
    startIndex = 1
  }
  const out: Holding[] = []
  for (let i = startIndex; i < lines.length; i += 1) {
    const parts = lines[i].split(',').map(v => v.trim())
    if (!parts.length) continue
    const ticker = parts[tickerIndex]?.toUpperCase()
    if (!ticker) continue
    const qtyRaw = parts[qtyIndex]?.replace(/,/g, '')
    const costRaw = parts[costIndex]?.replace(/,/g, '')
    const qty = Number(qtyRaw)
    const avgCost = Number(costRaw)
    if (!Number.isFinite(qty) || !Number.isFinite(avgCost)) continue
    out.push({ ticker, qty, avgCost })
  }
  return out
}

function toCsv(rows: Holding[]): string {
  const header = 'ticker,qty,avg_cost'
  const body = rows.map(r => `${r.ticker},${r.qty},${r.avgCost}`)
  return [header, ...body].join('\n')
}

function downloadFile(contents: string, filename: string, mime: string) {
  const blob = new Blob([contents], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function Portfolio() {
  const [holdings, setHoldings] = useState<Holding[]>([])
  const [meta, setMeta] = useState<StockMeta | null>(null)
  const [dataMap, setDataMap] = useState<Record<string, StockData>>({})

  const [loading, setLoading] = useState(false)
  const [refreshTick, setRefreshTick] = useState(0)

  useEffect(() => {
    try {
      const raw = localStorage.getItem(LS_KEY)
      if (raw) setHoldings(JSON.parse(raw))
    } catch {}
  }, [])

  useEffect(() => {
    try { localStorage.setItem(LS_KEY, JSON.stringify(holdings)) } catch {}
  }, [holdings])

  useEffect(() => {
    const run = async () => {
      try {
        setLoading(true)
        const m = await fetch(`${BASE}/data/meta.json`).then(r => r.json()) as StockMeta
        setMeta(m)
        const wanted = Array.from(new Set(holdings.map(h => h.ticker)))
        const results = await Promise.allSettled(wanted.map(t => fetch(`${BASE}/data/${t.replace(/\s+/g,'_')}.json`).then(r => r.json())))
        const ok = results.flatMap(r => r.status === 'fulfilled' ? [r.value as StockData] : [])
        const map: Record<string, StockData> = {}
        for (const s of ok) map[s.ticker] = s
        setDataMap(map)
      } catch (e) {
        console.error('Failed to load data', e)
      } finally {
        setLoading(false)
      }
    }
    if (holdings.length) run()
  }, [holdings, refreshTick])

  const [form, setForm] = useState<Holding>({ ticker: '', qty: 0, avgCost: 0 })

  const rows = useMemo(() => {
    return holdings.map(h => {
      const s = dataMap[h.ticker]
      const marketPrice = s?.price ?? null
      const price = marketPrice ?? (h.avgCost ?? null)
      const priceFallback = marketPrice == null && price != null
      const value = price != null ? price * h.qty : null
      const gain = price != null ? (price - h.avgCost) * h.qty : null
      const gainPct = price != null && h.avgCost ? ((price - h.avgCost) / h.avgCost) * 100 : null
      return { ...h, price, priceFallback, value, gain, gainPct, score: s?.score ?? null }
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
      const record: Holding = { ticker: t, qty: form.qty, avgCost: form.avgCost }
      if (ix >= 0) next[ix] = record
      else next.push(record)
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
    const isCsv = file.name.toLowerCase().endsWith('.csv') || file.type.includes('csv')
    const fr = new FileReader()
    fr.onload = () => {
      try {
        let parsed: Holding[] = []
        if (isCsv) {
          parsed = parseCsv(String(fr.result ?? ''))
        } else {
          const raw = JSON.parse(String(fr.result ?? '')) as Array<Partial<Holding>>
          parsed = (raw || []).map(x => normalizeHolding(x)).filter(Boolean) as Holding[]
        }
        if (parsed.length) setHoldings(parsed)
      } catch (err) {
        console.error('Import failed', err)
        alert('Failed to import portfolio file. Use CSV with ticker,qty,avg_cost or JSON array.')
      }
    }
    fr.readAsText(file)
    e.target.value = ''
  }

  function onExportJson() {
    downloadFile(JSON.stringify(holdings, null, 2), 'portfolio.json', 'application/json')
  }

  function onExportCsv() {
    downloadFile(toCsv(holdings), 'portfolio.csv', 'text/csv')
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
        <input type="file" accept={FILE_ACCEPT} onChange={onImport} />
        <button onClick={()=>setRefreshTick(x=>x+1)} disabled={loading || !holdings.length} style={btn}>{loading ? 'Refreshing...' : 'Refresh data'}</button>
        <button onClick={onExportJson} style={btn}>Export JSON</button>
        <button onClick={onExportCsv} style={btn}>Export CSV</button>
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
              <td style={td}>{fmt(r.price)}{r.priceFallback ? ' (est)' : ''}</td>
              <td style={td}>{fmt(r.value)}</td>
              <td style={{...td, color: (r.gain ?? 0) >= 0 ? 'green' : 'crimson'}}>{fmt(r.gain)}</td>
              <td style={td}>{pct(r.gainPct)}</td>
              <td style={{...td, color: scoreColor(r.score ?? 0)}}>{r.score ?? '--'}</td>
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
  if (n == null) return '--'
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(n)
}
function num(n?: number | null) {
  if (n == null) return '--'
  return new Intl.NumberFormat().format(n)
}
function pct(n?: number | null) {
  if (n == null) return '--'
  return `${n.toFixed(2)}%`
}
function scoreColor(v: number) {
  if (v >= 70) return 'green'
  if (v >= 50) return 'orange'
  return 'crimson'
}
