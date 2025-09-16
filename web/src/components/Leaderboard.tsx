"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockMeta, StockData } from '../types'
import { BASE } from '../base'

const META_CACHE_KEY = 'stockdash:meta'
const LIST_CACHE_KEY = 'stockdash:leaderboard'
const TICKER_CACHE_PREFIX = 'stockdash:ticker:'
const CACHE_TTL_MS = 10 * 60 * 1000

type CacheRecord<T> = {
  ts: number
  payload: T
}

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

export default function Leaderboard() {
  const [meta, setMeta] = useState<StockMeta | null>(null)
  const [items, setItems] = useState<StockData[]>([])
  const [q, setQ] = useState('')
  const [loading, setLoading] = useState(false)
  const [preset, setPreset] = useState<'NONE'|'HIGH_MOM'|'UNDERVALUED'|'LOW_ATR'>('NONE')
  const [onlyPriced, setOnlyPriced] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [usingCache, setUsingCache] = useState(false)
  const [newTicker, setNewTicker] = useState('')

  useEffect(() => {
    if (typeof window === 'undefined') return
    const cachedMeta = readCache<StockMeta>(META_CACHE_KEY)
    if (cachedMeta) setMeta(cachedMeta.payload)
    const cachedList = readCache<StockData[]>(LIST_CACHE_KEY)
    if (cachedList) setItems(cachedList.payload)
    fetchAll()
  }, [])

  async function fetchAll(force = false) {
    const cachedMeta = readCache<StockMeta>(META_CACHE_KEY)
    const cachedList = readCache<StockData[]>(LIST_CACHE_KEY)
    if (!force) {
      if (cachedMeta) setMeta(cachedMeta.payload)
      if (cachedList) setItems(cachedList.payload)
      const expectedTickers = cachedMeta?.payload?.tickers?.length ?? 0
      const cachedCount = cachedList && Array.isArray(cachedList.payload) ? cachedList.payload.length : 0
      if (cachedMeta && cachedList && isFresh(cachedMeta) && isFresh(cachedList) && cachedCount > 0 && (expectedTickers === 0 ? cachedCount > 1 : cachedCount >= expectedTickers)) {
        setUsingCache(true)
        return
      }
    }
    try {
      setLoading(true)
      setError(null)
      const metaJson = await fetch(`${BASE}/data/meta.json`).then(r => { if (!r.ok) throw new Error(`meta ${r.status}`); return r.json() as Promise<StockMeta> })
      setMeta(metaJson)
      writeCache(META_CACHE_KEY, metaJson)
      const tickers = metaJson.tickers ?? []
      const results = await Promise.allSettled(
        tickers.map(t => fetch(`${BASE}/data/${t.replace(/\s+/g,'_')}.json`).then(r => { if (!r.ok) throw new Error(`data ${r.status}`); return r.json() as Promise<StockData> }))
      )
      const ok = results.flatMap(r => r.status === 'fulfilled' ? [r.value as StockData] : [])
      setItems(ok)
      writeCache(LIST_CACHE_KEY, ok)
      ok.forEach(row => writeCache(`${TICKER_CACHE_PREFIX}${row.ticker}`, row))
      setUsingCache(false)
    } catch (e) {
      console.error('Failed to load data', e)
      setError('Failed to load leaderboard data.')
      if (cachedList) {
        setItems(prev => prev.length ? prev : cachedList.payload)
        setUsingCache(true)
      }
    } finally {
      setLoading(false)
    }
  }

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
        return typeof fy === 'number' && fy >= 0.08
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

  function openGithubIssueFor(action: 'ADD'|'REMOVE', symbol: string) {
    const t = (symbol || '').trim()
    if (!t) { alert('Enter a ticker symbol'); return }
    const title = `[${action}] ${t}`
    const body = 'Requested from dashboard UI'
    const url = `https://github.com/Kjaehr/STOCK_DASHBOARD/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`
    if (typeof window !== 'undefined') window.open(url, '_blank', 'noopener,noreferrer')
  }


  return (
    <section>
      <h1 style={{margin:'8px 0'}}>Leaderboard</h1>
      <div style={{display:'flex', gap:12, alignItems:'center', flexWrap:'wrap'}}>{/* Controls */}
        <input placeholder="Search ticker" value={q} onChange={e=>setQ(e.target.value)} style={{padding:'8px', border:'1px solid #ddd'}} />
        <button onClick={()=>fetchAll(true)} disabled={loading} style={btnMini}>{loading ? 'Refreshing...' : 'Refresh'}</button>
        <small style={{color:'#666'}}>Updated: {meta?.generated_at ?? '--'} {usingCache ? '(cache)' : ''}</small>
        <select value={preset} onChange={e=>setPreset(e.target.value as any)} style={{padding:'6px', border:'1px solid #ddd'}}>{/* Presets */}
          <option value="NONE">Presets</option>
          <option value="HIGH_MOM">High momentum</option>
          <option value="UNDERVALUED">Undervalued</option>
          <option value="LOW_ATR">Low ATR%</option>
        </select>
        <label style={{display:'flex', alignItems:'center', gap:6}}>{/* Only priced */}
          <input type="checkbox" checked={onlyPriced} onChange={e=>setOnlyPriced(e.target.checked)} />
          Only with price
        </label>
        <span style={{marginLeft:8, opacity:0.5}}>|</span>
        <input placeholder="Add/remove ticker (e.g. AAPL or NOVO-B.CO)" value={newTicker} onChange={e=>setNewTicker(e.target.value)} style={{padding:'8px', border:'1px solid #ddd'}} />
        <button onClick={()=>openGithubIssueFor('ADD', newTicker)} style={btnMini} title="Trigger GitHub Action via Issue">Add ticker</button>
        <button onClick={()=>openGithubIssueFor('REMOVE', newTicker)} style={btnMini} title="Trigger GitHub Action via Issue">Remove ticker</button>

      </div>
      {error ? <div style={alertWarn}>{error}</div> : null}
      <table style={{width:'100%', borderCollapse:'collapse', marginTop:12}}>{/* Table */}
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
              <td style={td}>
                {fmt(row.price)}{' '}
                {(row.price == null || (row.flags||[]).some(f => f.includes('no_price_data')))
                  ? <span style={badgeWarn} title="Yahoo price missing for this ticker">Missing data</span>
                  : null}
              </td>
              <td style={td}>{row.fund_points ?? 0}</td>
              <td style={td}>{row.tech_points ?? 0}</td>
              <td style={td}>{row.sent_points ?? 0}</td>
              <td style={td}><small>{(() => { const shown=(row.flags||[]).filter(f=>!String(f).includes('_fail')); return shown.length?shown.join(', '):'—' })()}</small></td>
              <td style={td}>
                <a href={`${BASE}/ticker/${encodeURIComponent(row.ticker)}`}>Details</a>
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
const alertWarn: React.CSSProperties = { marginTop:8, padding:'8px 12px', background:'#fff4e5', border:'1px solid #ffd8a8', borderRadius:8, color:'#8a6d3b' }

function scoreColor(n?: number) {
  const v = n ?? 0
  if (v >= 70) return 'green'
  if (v >= 50) return 'orange'
  return 'crimson'
}

function fmt(n?: number | null) {
  if (n == null) return '--'
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(n)
}
