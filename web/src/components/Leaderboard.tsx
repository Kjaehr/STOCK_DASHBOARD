"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockMeta, StockData } from '../types'
import { BASE } from '../base'

// UI components (shadcn/ui)
import { Input } from './ui/input'
import { Button } from './ui/button'
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from './ui/select'
import { Switch } from './ui/switch'
import { Badge } from './ui/badge'
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from './ui/table'

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
    <section className="space-y-3">
      <h1 className="text-xl font-semibold">Leaderboard</h1>
      <div className="flex flex-wrap items-center gap-3">{/* Controls */}
        <Input placeholder="Search ticker" value={q} onChange={e=>setQ(e.target.value)} className="w-48" />
        <Button variant="outline" size="sm" onClick={()=>fetchAll(true)} disabled={loading}>{loading ? 'Refreshing...' : 'Refresh'}</Button>
        <small className="text-xs text-muted-foreground">Updated: {meta?.generated_at ?? '--'} {usingCache ? '(cache)' : ''}</small>
        <Select value={preset} onValueChange={(v)=>setPreset(v as any)}>
          <SelectTrigger className="w-40"><SelectValue placeholder="Presets" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="NONE">Presets</SelectItem>
            <SelectItem value="HIGH_MOM">High momentum</SelectItem>
            <SelectItem value="UNDERVALUED">Undervalued</SelectItem>
            <SelectItem value="LOW_ATR">Low ATR%</SelectItem>
          </SelectContent>
        </Select>
        <div className="flex items-center gap-2">{/* Only priced */}
          <Switch checked={onlyPriced} onCheckedChange={setOnlyPriced} id="only-priced" />
          <label htmlFor="only-priced" className="text-sm text-muted-foreground">Only with price</label>
        </div>
        <span className="opacity-50">|</span>
        <Input placeholder="Add/remove ticker (e.g. AAPL or NOVO-B.CO)" value={newTicker} onChange={e=>setNewTicker(e.target.value)} className="w-80" />
        <Button size="sm" onClick={()=>openGithubIssueFor('ADD', newTicker)} title="Trigger GitHub Action via Issue">Add ticker</Button>
        <Button size="sm" variant="outline" onClick={()=>openGithubIssueFor('REMOVE', newTicker)} title="Trigger GitHub Action via Issue">Remove ticker</Button>
      </div>

      {error ? <div className="rounded-md border border-yellow-300 bg-yellow-50 px-3 py-2 text-yellow-800 dark:border-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200">{error}</div> : null}

      <Table>{/* Table */}
        <TableHeader>
          <TableRow>
            <TableHead>Ticker</TableHead>
            <TableHead>Score</TableHead>
            <TableHead>Price</TableHead>
            <TableHead>Fund</TableHead>
            <TableHead>Tech</TableHead>
            <TableHead>Sent</TableHead>
            <TableHead>Flags</TableHead>
            <TableHead></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filtered.map(row => (
            <TableRow key={row.ticker}>
              <TableCell>{row.ticker}</TableCell>
              <TableCell>
                <Badge variant="secondary" className={scoreClass(row.score)}>{row.score ?? 0}</Badge>
              </TableCell>
              <TableCell>
                {fmt(row.price)}{' '}
                {(row.price == null || (row.flags||[]).some(f => f.includes('no_price_data'))) ? (
                  <Badge variant="outline" className="border-yellow-300 bg-yellow-50 text-yellow-700 dark:border-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200">Missing data</Badge>
                ) : null}
              </TableCell>
              <TableCell>{row.fund_points ?? 0}</TableCell>
              <TableCell>{row.tech_points ?? 0}</TableCell>
              <TableCell>{row.sent_points ?? 0}</TableCell>
              <TableCell><small>{(() => { const shown=(row.flags||[]).filter(f=>!String(f).includes('_fail')); return shown.length?shown.join(', '):'—' })()}</small></TableCell>
              <TableCell className="space-x-2">
                <a className="underline underline-offset-2" href={`${BASE}/ticker/${encodeURIComponent(row.ticker)}`}>Details</a>
                <Button size="sm" onClick={()=>addToPortfolio(row.ticker, row.price)}>Add</Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  )
}


function scoreColor(n?: number) {
  const v = n ?? 0
  if (v >= 70) return 'green'
  if (v >= 50) return 'orange'
  return 'crimson'
}

function scoreClass(n?: number) {
  const v = n ?? 0
  if (v >= 70) return 'text-green-600'
  if (v >= 50) return 'text-orange-500'
  return 'text-red-600'
}

function fmt(n?: number | null) {
  if (n == null) return '--'
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(n)
}
