"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockMeta, StockData } from '../types'
import { BASE, DATA_BASE } from '../base'

// UI components (shadcn/ui)
import { Input } from './ui/input'
import { Button } from './ui/button'
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from './ui/select'
import { Switch } from './ui/switch'
import { Badge } from './ui/badge'
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from './ui/table'
import { ArrowUpDown } from 'lucide-react'
import * as Tooltip from '@radix-ui/react-tooltip'

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

  type SortKey = 'ticker'|'score'|'price'|'fund'|'tech'|'sent'
  const [sort, setSort] = useState<{ key: SortKey; dir: 'asc'|'desc' }>({ key: 'score', dir: 'desc' })
  function toggleSort(key: SortKey) {
    setSort(s => s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: key === 'ticker' ? 'asc' : 'desc' })
  }

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
      const metaJson = await fetch(`${DATA_BASE}/meta.json`).then(r => { if (!r.ok) throw new Error(`meta ${r.status}`); return r.json() as Promise<StockMeta> })
      setMeta(metaJson)
      writeCache(META_CACHE_KEY, metaJson)
      const tickers = metaJson.tickers ?? []
      const results = await Promise.allSettled(
        tickers.map(t => fetch(`${DATA_BASE}/${t.replace(/\s+/g,'_')}.json`).then(r => { if (!r.ok) throw new Error(`data ${r.status}`); return r.json() as Promise<StockData> }))
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
    return arr.sort((a,b) => {
      const dir = sort.dir === 'asc' ? 1 : -1
      const getVal = (x: StockData) => {
        switch (sort.key) {
          case 'ticker': return (x.ticker ?? '').toLowerCase()
          case 'price': return x.price ?? -Infinity
          case 'fund': return x.fund_points ?? -Infinity
          case 'tech': return x.tech_points ?? -Infinity
          case 'sent': return x.sent_points ?? -Infinity
          case 'score':
          default: return x.score ?? -Infinity
        }
      }
      const va = getVal(a); const vb = getVal(b)
      if (typeof va === 'string' && typeof vb === 'string') return va.localeCompare(vb) * dir
      const na = Number(va ?? -Infinity); const nb = Number(vb ?? -Infinity)
      if (na === nb) return 0
      return na > nb ? 1*dir : -1*dir
    })
  }, [items, q, preset, onlyPriced, sort])

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
      <Tooltip.Provider delayDuration={0}>
      <h1 className="text-xl font-semibold">Leaderboard</h1>
      <div className="flex flex-wrap items-center gap-3">{/* Controls */}
        <Input placeholder="Search ticker" value={q} onChange={e=>setQ(e.target.value)} className="w-48" />
        <Button variant="outline" size="sm" onClick={()=>fetchAll(true)} disabled={loading}>{loading ? 'Refreshing...' : 'Refresh'}</Button>
        <small className="text-xs text-muted-foreground">Updated: {meta?.generated_at ?? '--'} {usingCache ? '(cache)' : ''}</small>
        <Select value={preset} onValueChange={(v)=>setPreset(v as any)}>
          <SelectTrigger className="w-44"><SelectValue placeholder="Presets" /></SelectTrigger>
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

      <div className="rounded-md border bg-card shadow-sm overflow-auto max-h-[70vh]">
        <Table className="w-full text-sm min-w-[900px]">{/* Table */}
          <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <TableRow className="hover:bg-transparent">
              <TableHead className="whitespace-nowrap sticky left-0 z-20 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
                <button onClick={()=>toggleSort('ticker')} className="inline-flex items-center gap-1" title="Sort by ticker">Ticker <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="whitespace-nowrap">
                <button onClick={()=>toggleSort('score')} className="inline-flex items-center gap-1" title="Sort by total score">Score <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right whitespace-nowrap">
                <button onClick={()=>toggleSort('price')} className="inline-flex items-center gap-1" title="Sort by price">Price <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right whitespace-nowrap">
                <button onClick={()=>toggleSort('fund')} className="inline-flex items-center gap-1" title="Sort by fundamentals">Fund <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right whitespace-nowrap">
                <button onClick={()=>toggleSort('tech')} className="inline-flex items-center gap-1" title="Sort by technicals">Tech <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right whitespace-nowrap">
                <button onClick={()=>toggleSort('sent')} className="inline-flex items-center gap-1" title="Sort by sentiment">Sent <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="whitespace-nowrap">Flags</TableHead>
              <TableHead className="text-right" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map(row => (
              <TableRow key={row.ticker} className="odd:bg-muted/40 hover:bg-muted/50">
                <TableCell className="font-medium sticky left-0 z-10 bg-background/95">{row.ticker}</TableCell>
                <TableCell>
                  <Tooltip.Root>
                    <Tooltip.Trigger asChild>
                      <Badge className={scoreBadgeClass(row.score)} title={`Fund ${row.fund_points ?? '--'} / Tech ${row.tech_points ?? '--'} / Sent ${row.sent_points ?? '--'}`}>{row.score ?? 0}</Badge>
                    </Tooltip.Trigger>
                    <Tooltip.Portal>
                      <Tooltip.Content side="top" sideOffset={6} className="z-50 rounded border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-md">
                        Fund {row.fund_points ?? '--'} / Tech {row.tech_points ?? '--'} / Sent {row.sent_points ?? '--'}
                        <Tooltip.Arrow className="fill-border" />
                      </Tooltip.Content>
                    </Tooltip.Portal>
                  </Tooltip.Root>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {fmt(row.price)}{' '}
                  {(row.price == null || (row.flags||[]).some(f => f.includes('no_price_data'))) ? (
                    <Badge variant="outline" className="ml-2 border-yellow-300 bg-yellow-50 text-yellow-700 dark:border-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200">Missing data</Badge>
                  ) : null}
                </TableCell>
                <TableCell className="text-right tabular-nums">{row.fund_points ?? 0}</TableCell>
                <TableCell className="text-right tabular-nums">{row.tech_points ?? 0}</TableCell>
                <TableCell className="text-right tabular-nums">{row.sent_points ?? 0}</TableCell>
                <TableCell className="max-w-[220px] truncate">
                  {(() => {
                    const shown=(row.flags||[]).filter(f=>!String(f).includes('_fail'))
                    if (!shown.length) return '—'
                    return (
                      <Tooltip.Root>
                        <Tooltip.Trigger asChild>
                          <span className="flex flex-wrap gap-1">
                            {shown.slice(0,4).map(f => (
                              <Badge key={String(f)} variant="outline" className="font-normal">{String(f)}</Badge>
                            ))}
                            {shown.length > 4 ? <span className="text-xs text-muted-foreground">+{shown.length-4} more</span> : null}
                          </span>
                        </Tooltip.Trigger>
                        <Tooltip.Portal>
                          <Tooltip.Content side="top" sideOffset={6} className="z-50 rounded border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-md max-w-[280px]">
                            {shown.join(', ')}
                            <Tooltip.Arrow className="fill-border" />
                          </Tooltip.Content>
                        </Tooltip.Portal>
                      </Tooltip.Root>
                    )
                  })()}
                </TableCell>
                <TableCell className="text-right space-x-2 whitespace-nowrap">
                  <a className="underline underline-offset-2" href={`${BASE}/ticker?id=${encodeURIComponent(row.ticker)}`}>Details</a>
                  <Button size="sm" onClick={()=>addToPortfolio(row.ticker, row.price)}>Add</Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>Showing {filtered.length} of {items.length}</span>
        <span>Average score: {filtered.length ? (filtered.reduce((a,r)=>a + (r.score ?? 0), 0) / filtered.length).toFixed(1) : '--'}</span>
      </div>
      </Tooltip.Provider>
    </section>
  )
}


function scoreBadgeClass(n?: number) {
  const v = n ?? 0
  if (v >= 70) return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
  if (v >= 50) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-200'
  return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
}



function fmt(n?: number | null) {
  if (n == null) return '--'
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(n)
}
