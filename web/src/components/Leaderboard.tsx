"use client"
import { useEffect, useMemo, useState, useRef } from 'react'
import type { StockMeta, StockData } from '../types'
import { BASE, DATA_BASE } from '../base'
import { useRouter, usePathname, useSearchParams } from 'next/navigation'

// UI components (shadcn/ui)
import { Input } from './ui/input'
import { Button } from './ui/button'

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

function clearCaches() {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(META_CACHE_KEY)
    window.localStorage.removeItem(LIST_CACHE_KEY)
    // Remove per-ticker caches
    const keys: string[] = []
    for (let i = 0; i < window.localStorage.length; i++) {
      const k = window.localStorage.key(i)
      if (k && k.startsWith(TICKER_CACHE_PREFIX)) keys.push(k)
    }
    keys.forEach(k => window.localStorage.removeItem(k))
  } catch (e) {
    console.warn('cache clear failed', e)
  }
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
  const [preset, setPreset] = useState<'NONE'|'HIGH_MOM'|'UNDERVALUED'|'LOW_ATR'>(() => {
    const p = (typeof window !== 'undefined') ? new URLSearchParams(window.location.search).get('p') as ('HIGH_MOM'|'UNDERVALUED'|'LOW_ATR'|null) : null
    return (p ?? 'NONE') as 'NONE'|'HIGH_MOM'|'UNDERVALUED'|'LOW_ATR'
  })
  const [onlyPriced, setOnlyPriced] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [usingCache, setUsingCache] = useState(false)

  // Routing/URL state
  const router = useRouter()
  const pathname = usePathname()
  const sp = useSearchParams()


  // Active preset is the component state; URL is just a reflection
  const activePreset: 'NONE'|'HIGH_MOM'|'UNDERVALUED'|'LOW_ATR' = preset

  type SortKey = 'ticker'|'score'|'price'|'fund'|'tech'|'sent'
  const [sort, setSort] = useState<{ key: SortKey; dir: 'asc'|'desc' }>({ key: 'score', dir: 'desc' })
  function toggleSort(key: SortKey) {
    setSort(s => s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: key === 'ticker' ? 'asc' : 'desc' })
  }
  // Helper to update URL immediately (avoid race with effects)
  function replaceUrlFromState(statePreset: 'NONE'|'HIGH_MOM'|'UNDERVALUED'|'LOW_ATR' = preset) {
    if (!router || !pathname) return
    const params = new URLSearchParams(sp?.toString() || '')
    // keep current state
    if (q) params.set('q', q); else params.delete('q')
    if (statePreset && statePreset !== 'NONE') params.set('p', statePreset); else params.delete('p')
    if (onlyPriced) params.set('op', '1'); else params.delete('op')
    const isDefaultSort = sort.key === 'score' && sort.dir === 'desc'
    if (!isDefaultSort) { params.set('s', sort.key); params.set('d', sort.dir) } else { params.delete('s'); params.delete('d') }
    const next = `${pathname}${params.toString() ? `?${params.toString()}` : ''}`
    const current = `${pathname}${sp && sp.toString() ? `?${sp.toString()}` : ''}`
    if (next !== current) router.replace(next, { scroll: false })
  }

  function handlePreset(p: 'NONE'|'HIGH_MOM'|'UNDERVALUED'|'LOW_ATR') {
    setPreset(p)
    // update URL immediately to make state sticky
    replaceUrlFromState(p)
  }


  // Guard to avoid writing URL before we've initialized from it
  const readyRef = useRef(false)



  const [newTicker, setNewTicker] = useState('')
  const searchRef = useRef<HTMLInputElement>(null)

  // Keyboard shortcut: "/" focuses search input
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === '/' && !e.metaKey && !e.altKey) {
        const target = e.target as HTMLElement | null
        const tag = target?.tagName?.toLowerCase()
        if (tag !== 'input' && tag !== 'textarea' && !(target as any)?.isContentEditable) {
          e.preventDefault()
          searchRef.current?.focus()
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  // Read state from URL on navigation/back/forward
  useEffect(() => {
    const params = new URLSearchParams(sp?.toString() || '')
    const urlQ = params.get('q') ?? null
    const urlP = params.get('p') as 'HIGH_MOM'|'UNDERVALUED'|'LOW_ATR'|null
    const hasP = params.has('p')
    const urlOP = params.get('op') === '1'
    const urlS = (params.get('s') as any) || 'score'
    const urlD = (params.get('d') as any) || 'desc'
    if (urlQ !== null && urlQ !== q) setQ(urlQ)
    // Only update preset from URL if the 'p' param is explicitly present
    if (hasP) {
      const nextPreset = (urlP ?? 'NONE') as 'NONE'|'HIGH_MOM'|'UNDERVALUED'|'LOW_ATR'
      if (nextPreset !== preset) setPreset(nextPreset)
    }
    if (urlOP !== onlyPriced) setOnlyPriced(urlOP)
    if (urlS !== sort.key || urlD !== sort.dir) setSort({ key: urlS as any, dir: urlD as any })
    // Mark as ready so URL-writer effect can run
    readyRef.current = true
  }, [sp])

  // Write state to URL when filters change (only after initial URL sync)
  useEffect(() => {
    if (!router || !pathname) return
    if (!readyRef.current) return
    const params = new URLSearchParams(sp?.toString() || '')
    if (q) params.set('q', q); else params.delete('q')
    if (preset && preset !== 'NONE') params.set('p', preset); else params.delete('p')
    if (onlyPriced) params.set('op', '1'); else params.delete('op')
    const isDefaultSort = sort.key === 'score' && sort.dir === 'desc'
    if (!isDefaultSort) { params.set('s', sort.key); params.set('d', sort.dir) } else { params.delete('s'); params.delete('d') }
    const next = `${pathname}${params.toString() ? `?${params.toString()}` : ''}`
    const current = `${pathname}${sp && sp.toString() ? `?${sp.toString()}` : ''}`
    if (next !== current) router.replace(next, { scroll: false })
  }, [q, preset, onlyPriced, sort, pathname, router])


  useEffect(() => {
    if (typeof window === 'undefined') return
    const cachedMeta = readCache<StockMeta>(META_CACHE_KEY)
    if (cachedMeta) setMeta(cachedMeta.payload)
    const cachedList = readCache<StockData[]>(LIST_CACHE_KEY)
    if (cachedList) setItems(cachedList.payload)
    fetchAll()
  }, [])

  async function fetchAll(force = false) {
    if (force) clearCaches()
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
      const bust = `?t=${Date.now()}`
      const metaJson = await fetch(`${DATA_BASE}/meta.json${force ? bust : ''}`, { cache: force ? 'no-store' : 'default' }).then(r => { if (!r.ok) throw new Error(`meta ${r.status}`); return r.json() as Promise<StockMeta> })
      const tickers = metaJson.tickers ?? []

      if (force) {
        // Live refresh via runtime screener API in batches (≤10 per request)
        const chunks: string[][] = []
        for (let i = 0; i < tickers.length; i += 10) chunks.push(tickers.slice(i, i + 10))
        const calls = chunks.map(async chunk => {
          const qs = encodeURIComponent(chunk.join(','))
          const resp = await fetch(`/api/screener?tickers=${qs}&refresh=1`, { cache: 'no-store' }).then(r => r.json() as Promise<{ generated_at: string; items: any[] }>)
          return resp
        })
        const responses = await Promise.allSettled(calls)
        const itemsLive: StockData[] = responses.flatMap(r => {
          if (r.status !== 'fulfilled') return []
          return (r.value.items || []).map((it: any) => ({
            ticker: it.ticker,
            price: it.price ?? null,
            score: it.score ?? null,
            fund_points: it.fund_points ?? null,
            tech_points: it.tech_points ?? null,
            sent_points: it.sent_points ?? null,
            flags: it.flags ?? [],
            technicals: it.technicals ?? {},
            sentiment: it.sentiment ?? {},
          } as StockData))
        })
        setItems(itemsLive)
        // Update meta with screener timestamp, keep tickers list
        setMeta({ generated_at: new Date().toISOString(), tickers } as any)
        writeCache(LIST_CACHE_KEY, itemsLive)
        itemsLive.forEach(row => writeCache(`${TICKER_CACHE_PREFIX}${row.ticker}`, row))
        writeCache(META_CACHE_KEY, { generated_at: new Date().toISOString(), tickers } as any)
        setUsingCache(false)
      } else {
        // Default: fetch from Supabase-backed JSONs (fast path)
        setMeta(metaJson)
        writeCache(META_CACHE_KEY, metaJson)
        const results = await Promise.allSettled(
          tickers.map(t => fetch(`${DATA_BASE}/${t.replace(/\s+/g,'_')}.json${force ? bust : ''}`, { cache: force ? 'no-store' : 'default' }).then(r => { if (!r.ok) throw new Error(`data ${r.status}`); return r.json() as Promise<StockData> }))
        )
        const ok = results.flatMap(r => r.status === 'fulfilled' ? [r.value as StockData] : [])
        setItems(ok)
        writeCache(LIST_CACHE_KEY, ok)
        ok.forEach(row => writeCache(`${TICKER_CACHE_PREFIX}${row.ticker}`, row))
        setUsingCache(false)
      }
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
    if (activePreset === 'HIGH_MOM') {
      arr = arr.filter(x => {
        const rsi = (x as any)?.technicals?.rsi
        return typeof rsi === 'number' && rsi > 60
      })
    } else if (activePreset === 'UNDERVALUED') {
      arr = arr.filter(x => {
        const fy = (x as any)?.fundamentals?.fcf_yield
        return typeof fy === 'number' && fy >= 0.08
      })
    } else if (activePreset === 'LOW_ATR') {
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
        <Input ref={searchRef} placeholder="Search ticker (/)" value={q} onChange={e=>setQ(e.target.value)} className="w-48"
          onKeyDown={(e)=>{
            if (e.key === 'Enter') {
              const s = q.trim().toLowerCase()
              const exact = filtered.find(x => (x.ticker||'').toLowerCase() === s)?.ticker
              const first = filtered[0]?.ticker
              const go = exact || first
              if (go) window.location.href = `${BASE}/ticker?id=${encodeURIComponent(go)}`
            }
          }}
        />
        <span className="text-xs text-muted-foreground hidden sm:inline">Shortcuts: "/" focus, Enter opens</span>

        <Button variant="outline" size="sm" onClick={()=>fetchAll(true)} disabled={loading}>{loading ? 'Refreshing...' : 'Refresh'}</Button>
        <small className="text-xs text-muted-foreground">Updated: {meta?.generated_at ?? '--'} {usingCache ? '(cache)' : ''}</small>
        <Badge variant="outline" className="font-normal" title="DATA_BASE endpoint">Endpoint: {DATA_BASE}</Badge>
        <div className="flex items-center gap-1">
          {(['NONE','HIGH_MOM','UNDERVALUED','LOW_ATR'] as const).map(p => (
            <Button key={p} size="sm" variant={activePreset===p? 'default':'outline'} aria-pressed={activePreset===p} onClick={()=>handlePreset(p)}>
              {p==='NONE'?'All':p==='HIGH_MOM'?'High mom':p==='UNDERVALUED'?'Undervalued':'Low ATR%'}
            </Button>
          ))}
        </div>
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

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-md border bg-card/60 backdrop-blur p-3">
          <div className="text-xs text-muted-foreground">Tickers</div>
          <div className="text-lg font-semibold">{items.length}</div>
        </div>
        <div className="rounded-md border bg-card/60 backdrop-blur p-3">
          <div className="text-xs text-muted-foreground">Avg score</div>
          <div className="text-lg font-semibold">{filtered.length ? (filtered.reduce((a,r)=>a + (r.score ?? 0), 0) / filtered.length).toFixed(1) : '--'}</div>
        </div>
        <div className="rounded-md border bg-card/60 backdrop-blur p-3">
          <div className="text-xs text-muted-foreground">Updated</div>
          <div className="text-lg font-semibold">{meta?.generated_at ?? '--'}</div>
        </div>
        <div className="rounded-md border bg-card/60 backdrop-blur p-3">
          <div className="text-xs text-muted-foreground">Cache</div>
          <div className="text-lg font-semibold">{usingCache ? 'Cache' : 'Live'}</div>
        </div>
      </div>

      {/* Empty state when no results */}
      {!loading && filtered.length === 0 ? (
        <div className="rounded-md border bg-card/60 backdrop-blur p-6">
          <h3 className="text-sm font-medium mb-1">No results</h3>
          <p className="text-sm text-muted-foreground mb-3">Try adjusting search or filters.</p>
          <Button size="sm" variant="outline" onClick={()=>{ setQ(''); setPreset('NONE'); setOnlyPriced(false); setSort({ key:'score', dir:'desc' }) }}>Reset filters</Button>
        </div>
      ) : null}

      <div className="rounded-md border bg-card shadow-sm overflow-auto max-h-[70vh]">
        <Table className="w-full text-sm min-w-[900px]">{/* Table */}
          <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <TableRow className="hover:bg-transparent">
              <TableHead className="whitespace-nowrap sticky left-0 z-20 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-r">
                <button onClick={()=>toggleSort('ticker')} className="inline-flex items-center gap-1" title="Sort by ticker">Ticker <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="whitespace-nowrap">
                <button onClick={()=>toggleSort('score')} className="inline-flex items-center gap-1" title="Sort by total score">Score <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right whitespace-nowrap">
                <button onClick={()=>toggleSort('price')} className="inline-flex items-center gap-1" title="Sort by price">Price <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right whitespace-nowrap">RSI</TableHead>
              <TableHead className="text-right whitespace-nowrap">ATR%</TableHead>
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
            {loading ? (
              Array.from({ length: 8 }).map((_, i) => (
                <TableRow key={`sk-${i}`} className="odd:bg-muted/40 animate-pulse">
                  <TableCell className="font-medium sticky left-0 z-10 bg-background/80 border-r"><div className="h-4 w-16 bg-muted rounded" /></TableCell>
                  <TableCell><div className="h-4 w-10 bg-muted rounded" /></TableCell>
                  <TableCell className="text-right"><div className="ml-auto h-4 w-14 bg-muted rounded" /></TableCell>
                  <TableCell className="text-right"><div className="ml-auto h-4 w-10 bg-muted rounded" /></TableCell>
                  <TableCell className="text-right"><div className="ml-auto h-4 w-10 bg-muted rounded" /></TableCell>
                  <TableCell className="text-right"><div className="ml-auto h-4 w-10 bg-muted rounded" /></TableCell>
                  <TableCell className="text-right"><div className="ml-auto h-4 w-10 bg-muted rounded" /></TableCell>
                  <TableCell className="text-right"><div className="ml-auto h-4 w-10 bg-muted rounded" /></TableCell>
                  <TableCell><div className="h-4 w-40 bg-muted rounded" /></TableCell>
                  <TableCell className="text-right"><div className="ml-auto h-7 w-24 bg-muted rounded" /></TableCell>
                </TableRow>
              ))
            ) : (
              filtered.map(row => (
                <TableRow key={row.ticker} className="odd:bg-muted/40 hover:bg-muted/50">
                  <TableCell className="font-medium sticky left-0 z-10 bg-background/95 border-r">{row.ticker}</TableCell>
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
                  <TableCell className="text-right tabular-nums">
                    {fmt((row as any)?.technicals?.rsi)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {(row as any)?.technicals?.atr_pct != null ? `${fmt((row as any)?.technicals?.atr_pct)}%` : '--'}
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
              ))
            )}
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
