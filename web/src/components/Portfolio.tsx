"use client"
import { useEffect, useMemo, useState } from 'react'
import type { StockData } from '../types'
import { BASE } from '../base'

// UI components (shadcn/ui)
import { Input } from './ui/input'
import { Button } from './ui/button'
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from './ui/table'
import { Badge } from './ui/badge'
import { ArrowUpDown } from 'lucide-react'

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
  const [dataMap, setDataMap] = useState<Record<string, StockData>>({})

  const [loading, setLoading] = useState(false)
  const [refreshTick, setRefreshTick] = useState(0)

  type SortKey = 'ticker'|'qty'|'avgCost'|'price'|'value'|'gain'|'gainPct'|'score'
  const [sort, setSort] = useState<{ key: SortKey; dir: 'asc'|'desc' }>({ key: 'value', dir: 'desc' })
  function toggleSort(key: SortKey) {
    setSort(s => s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: key === 'ticker' ? 'asc' : 'desc' })
  }

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

  const sortedRows = useMemo(() => {
    const dir = sort.dir === 'asc' ? 1 : -1
    const getVal = (x: any) => {
      switch (sort.key) {
        case 'ticker': return (x.ticker ?? '').toLowerCase()
        case 'qty': return x.qty ?? -Infinity
        case 'avgCost': return x.avgCost ?? -Infinity
        case 'price': return x.price ?? -Infinity
        case 'value': return x.value ?? -Infinity
        case 'gain': return x.gain ?? -Infinity
        case 'gainPct': return x.gainPct ?? -Infinity
        case 'score': default: return x.score ?? -Infinity
      }
    }
    return [...rows].sort((a,b) => {
      const va = getVal(a); const vb = getVal(b)
      if (typeof va === 'string' && typeof vb === 'string') return va.localeCompare(vb) * dir
      const na = Number(va ?? -Infinity); const nb = Number(vb ?? -Infinity)
      if (na === nb) return 0
      return na > nb ? 1*dir : -1*dir
    })
  }, [rows, sort])

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
    <section className="space-y-3">
      <h1 className="text-xl font-semibold">Portfolio</h1>
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-sm text-muted-foreground">Ticker</label>
          <Input value={form.ticker} onChange={e=>setForm({...form, ticker:e.target.value})} placeholder="e.g., APLD" className="w-32" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-sm text-muted-foreground">Qty</label>
          <Input type="number" value={form.qty} onChange={e=>setForm({...form, qty:Number(e.target.value)})} className="w-28" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-sm text-muted-foreground">Avg cost</label>
          <Input type="number" value={form.avgCost} onChange={e=>setForm({...form, avgCost:Number(e.target.value)})} className="w-28" />
        </div>
        <Button onClick={addHolding}>Add/Update</Button>
        <input type="file" accept={FILE_ACCEPT} onChange={onImport} className="text-sm" />
        <Button onClick={()=>setRefreshTick(x=>x+1)} disabled={loading || !holdings.length}>{loading ? 'Refreshing...' : 'Refresh data'}</Button>
        <Button onClick={onExportJson} variant="outline">Export JSON</Button>
        <Button onClick={onExportCsv} variant="outline">Export CSV</Button>
      </div>

      <div>
        <strong>Total value:</strong> {fmt(totals.totalValue)}
      </div>

      <div className="rounded-md border bg-card shadow-sm overflow-auto max-h-[70vh]">
        <Table className="w-full text-sm min-w-[900px]">
          <TableHeader className="sticky top-0 z-10 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <TableRow className="hover:bg-transparent">
              <TableHead className="sticky left-0 z-20 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
                <button onClick={()=>toggleSort('ticker')} className="inline-flex items-center gap-1">Ticker <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right">
                <button onClick={()=>toggleSort('qty')} className="inline-flex items-center gap-1">Qty <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right">
                <button onClick={()=>toggleSort('avgCost')} className="inline-flex items-center gap-1">Avg cost <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right">
                <button onClick={()=>toggleSort('price')} className="inline-flex items-center gap-1">Price <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right">
                <button onClick={()=>toggleSort('value')} className="inline-flex items-center gap-1">Value <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right">
                <button onClick={()=>toggleSort('gain')} className="inline-flex items-center gap-1">Gain <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right">
                <button onClick={()=>toggleSort('gainPct')} className="inline-flex items-center gap-1">Gain % <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right">
                <button onClick={()=>toggleSort('score')} className="inline-flex items-center gap-1">Score <ArrowUpDown className="h-3.5 w-3.5 opacity-60" /></button>
              </TableHead>
              <TableHead className="text-right" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedRows.map(r => (
              <TableRow key={r.ticker} className="odd:bg-muted/40 hover:bg-muted/50">
                <TableCell className="font-medium sticky left-0 z-10 bg-background/95">{r.ticker}</TableCell>
                <TableCell className="text-right tabular-nums">{num(r.qty)}</TableCell>
                <TableCell className="text-right tabular-nums">{fmt(r.avgCost)}</TableCell>
                <TableCell className="text-right tabular-nums">{fmt(r.price)}{r.priceFallback ? ' (est)' : ''}</TableCell>
                <TableCell className="text-right tabular-nums">{fmt(r.value)}</TableCell>
                <TableCell className={"text-right tabular-nums " + ((r.gain ?? 0) >= 0 ? 'text-green-600' : 'text-red-600')}>{fmt(r.gain)}</TableCell>
                <TableCell className="text-right tabular-nums">{pct(r.gainPct)}</TableCell>
                <TableCell className="text-right"><Badge className={scoreBadgeClass(r.score ?? 0)}>{r.score ?? '--'}</Badge></TableCell>
                <TableCell className="text-right"><Button variant="outline" size="sm" onClick={()=>removeHolding(r.ticker)}>Remove</Button></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </section>
  )
}

function scoreBadgeClass(n?: number | null) {
  const v = (n ?? 0) as number
  if (v >= 70) return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
  if (v >= 50) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-200'
  return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
}


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