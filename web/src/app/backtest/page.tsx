"use client"
import React from 'react'
import { LineChart } from '../../components/Charts'

function Num({label, value, onChange, step=0.1}:{label:string; value:number; onChange:(v:number)=>void; step?:number}){
  return (
    <label className="flex items-center gap-2 text-sm">
      <span className="w-40 opacity-80">{label}</span>
      <input type="number" step={step} value={value} onChange={e=>onChange(Number(e.target.value))} className="w-28 rounded border bg-background px-2 py-1" />
    </label>
  )
}

export default function BacktestPage(){
  const [tickers, setTickers] = React.useState<string>('NVO, SPY, MSFT')
  const [lookbackYears, setLookbackYears] = React.useState<number>(2)
  const [params, setParams] = React.useState<any>({
    sma20:{enabled:true, k_atr:1},
    sma50:{enabled:true, k_atr:1},
    breakout:{enabled:true, lookback:60, k_atr:0.5},
    stops:{ma:'sma50', atr_mult:1},
    targets:{multiples:[1.5,2.5]},
    filters:{trendUp:true, rsiRange:[45,65]},
  })
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string|undefined>()
  const [run, setRun] = React.useState<any>(null)
  const [runId, setRunId] = React.useState<string|undefined>()
  const [trades, setTrades] = React.useState<any[]>([])
  const [asyncMode, setAsyncMode] = React.useState(false)
  const [history, setHistory] = React.useState<any[]>([])
  const [historyTotal, setHistoryTotal] = React.useState(0)
  const [historyPage, setHistoryPage] = React.useState(1)
  const [historyPageSize, setHistoryPageSize] = React.useState(10)
  const [tickerFilter, setTickerFilter] = React.useState('')
  const [dateFrom, setDateFrom] = React.useState<string>('')
  const [dateTo, setDateTo] = React.useState<string>('')
  const [presets, setPresets] = React.useState<any[]>([])
  const [presetName, setPresetName] = React.useState('')

  async function loadHistory(){
    try{
      const qs = new URLSearchParams()
      qs.set('page', String(historyPage))
      qs.set('pageSize', String(historyPageSize))
      if (tickerFilter.trim()) qs.set('ticker', tickerFilter.trim().toUpperCase())
      if (dateFrom) qs.set('from', dateFrom)
      if (dateTo) qs.set('to', dateTo)
      const r = await fetch(`/api/backtest/list?${qs.toString()}`, { cache:'no-store' })
      const j = await r.json()
      if(r.ok){ setHistory(j?.items||[]); setHistoryTotal(Number(j?.total||0)) }
    }catch{}
  }
  React.useEffect(()=>{ loadHistory() }, [historyPage, historyPageSize])

  async function loadRun(id:string){
    setLoading(true); setError(undefined)
    try{
      const r = await fetch(`/api/backtest?id=${encodeURIComponent(id)}`, { cache:'no-store' })
      const j = await r.json()
      if(!r.ok || j?.error) throw new Error(j?.error || `HTTP ${r.status}`)
      const perTicker = Array.isArray(j?.run?.summary?.perTicker) ? j.run.summary.perTicker : []
      const summary = j?.run?.summary || {}
      const equityCurve = j?.run?.equity || []
      setRun({ summary, perTicker, equityCurve })
      setRunId(j?.run?.id)
      setTrades(Array.isArray(j?.trades)? j.trades: [])
    }catch(e:any){ setError(e?.message || String(e)) }
    finally{ setLoading(false) }
  }

  async function runBacktest(){
    setLoading(true); setError(undefined); setRun(null); setRunId(undefined); setTrades([])
    try{
      const body: any = { tickers: tickers.split(',').map(t=>t.trim()).filter(Boolean), lookbackYears, params }
      if (asyncMode) body.mode = 'async'
      const r = await fetch('/api/backtest', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body) })
      const j = await r.json()
      if(!r.ok || j?.error){ throw new Error(j?.error || `HTTP ${r.status}`) }
      setRun(j); setRunId(j?.run_id); loadHistory()
      if (j?.run_id){
        // fetch trades/details
        try { await loadRun(j.run_id) } catch {}
        if (asyncMode){
          // Poll status until completed (lightweight)
          const id = j.run_id
          let tries = 0
          const iv = setInterval(async()=>{
            tries++
            try{
              const rr = await fetch(`/api/backtest?id=${encodeURIComponent(id)}`, { cache:'no-store' })
              const jj = await rr.json()
              if (jj?.run?.status === 'completed') { setTrades(jj.trades||[]); clearInterval(iv) }
              if (tries > 20) clearInterval(iv)
            }catch{}
          }, 1500)
        }
      }
    }catch(e:any){ setError(e?.message || String(e)) }
    finally{ setLoading(false) }
  }
  async function loadPresets(){
    try{
      const r = await fetch('/api/backtest/presets', { cache:'no-store' })
      const j = await r.json()
      if (r.ok) setPresets(j?.items||[])
    }catch{}
  }
  React.useEffect(()=>{ loadPresets() }, [])

  async function savePreset(){
    try{
      if (!presetName.trim()) return
      const r = await fetch('/api/backtest/presets', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({ name: presetName.trim(), params }) })
      const j = await r.json()
      if (!r.ok || j?.error) throw new Error(j?.error || `HTTP ${r.status}`)
      setPresetName(''); loadPresets()
    }catch(e:any){ setError(e?.message || String(e)) }
  }

  function applyPreset(id: string){
    const p = presets.find(x=>x.id===id)
    if (p && p.params) setParams(p.params)
  }

  const [presetA, setPresetA] = React.useState<string>('')
  const [presetB, setPresetB] = React.useState<string>('')
  const [compare, setCompare] = React.useState<any[]>([])
  async function runCompare(){
    const picks = [presetA, presetB].filter(Boolean)
    const results: any[] = []
    for (const id of picks){
      const p = presets.find(x=>x.id===id)
      if (!p) continue
      const body = { tickers: tickers.split(',').map(t=>t.trim()).filter(Boolean), lookbackYears, params: p.params }
      const r = await fetch('/api/backtest', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body) })
      const j = await r.json()
      if (r.ok && !j?.error) results.push({ name: p.name, summary: j.summary })
    }
    setCompare(results)
  }


  const labels = (run?.equityCurve||[]).map((p:any)=>String(p.i))
  const eq = (run?.equityCurve||[]).map((p:any)=>p.eq)

  return (
    <div className="p-4 md:p-6 max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Backtest</h1>
        <p className="text-sm opacity-80">Kør backtests over 1–5 år og justér parametre. Resultat gemmes i Supabase.</p>
      </div>


          <div className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={asyncMode} onChange={e=>setAsyncMode(e.target.checked)} />
            <span>Async mode (poll efter status)</span>
          </div>

          <div className="mt-3 space-y-2">
            <div className="text-xs font-medium opacity-80">Presets</div>
            <div className="flex items-center gap-2">
              <input value={presetName} onChange={e=>setPresetName(e.target.value)} placeholder="Navn..." className="rounded border bg-background px-2 py-1"/>
              <button onClick={savePreset} className="rounded border px-3 py-1.5 text-sm">Gem preset</button>
            </div>
            <div className="flex items-center gap-2">
              <select onChange={e=>applyPreset(e.target.value)} className="rounded border bg-background px-2 py-1">
                <option value="">Indlæs preset...</option>
                {presets.map(p=> <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <span className="text-xs opacity-60">Loader overskriver felterne ovenfor</span>
            </div>
            <div className="flex items-center gap-2">
              <select value={presetA} onChange={e=>setPresetA(e.target.value)} className="rounded border bg-background px-2 py-1">
                <option value="">A</option>
                {presets.map(p=> <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <select value={presetB} onChange={e=>setPresetB(e.target.value)} className="rounded border bg-background px-2 py-1">
                <option value="">B</option>
                {presets.map(p=> <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <button onClick={runCompare} className="rounded border px-3 py-1.5 text-sm">Kør compare</button>
            </div>
            {compare?.length ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left opacity-70"><tr><th className="py-1">Preset</th><th>Trades</th><th>Hit</th><th>Exp (R)</th></tr></thead>
                  <tbody>
                    {compare.map((c:any)=> (
                      <tr key={c.name} className="border-t"><td className="py-1">{c.name}</td><td>{c.summary?.trades ?? '-'}</td><td>{c.summary?.hit_rate != null ? Math.round(c.summary.hit_rate*100)+'%' : '-'}</td><td>{c.summary?.expectancy_R ?? '-'}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-3 rounded-lg border p-4">
          <label className="text-sm">
            <div className="opacity-80 mb-1">Tickers (op til 20)</div>
            <input value={tickers} onChange={e=>setTickers(e.target.value)} placeholder="NVO, SPY, MSFT" className="w-full rounded border bg-background px-2 py-1"/>
          </label>
          <label className="text-sm">
            <div className="opacity-80 mb-1">Lookback (år)</div>
            <select value={lookbackYears} onChange={e=>setLookbackYears(Number(e.target.value))} className="w-32 rounded border bg-background px-2 py-1">
              {[1,2,3,4,5].map(n=> <option key={n} value={n}>{n} år</option> )}

      {trades?.length ? (
        <div className="rounded-lg border p-4">
          <div className="text-sm font-medium mb-2">Trades (seneste {Math.min(trades.length, 200)})</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left opacity-70">
                <tr>
                  <th className="py-1">Ticker</th><th>Entry</th><th>Exit</th><th>Pris ind</th><th>Pris ud</th><th>R</th><th>%</th><th>Bars</th><th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {trades.slice(0,200).map((t:any, i:number)=> (
                  <tr key={i} className="border-t">
                    <td className="py-1">{t.ticker}</td>
                    <td>{t.entry_date}</td>
                    <td>{t.exit_date}</td>
                    <td>{t.entry_price}</td>
                    <td>{t.exit_price}</td>
                    <td>{t.r}</td>
                    <td>{t.pnl_pct}</td>
                    <td>{t.bars_held}</td>
                    <td>{t.exit_reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}


            </select>
          </label>

          <div className="mt-2 text-xs font-medium opacity-80">Strategi</div>
          <div className="grid grid-cols-2 gap-3">
            <Num label="SMA20 k·ATR" value={params.sma20.k_atr} onChange={v=>setParams((p:any)=>({...p, sma20:{...p.sma20, k_atr:v}}))} />
            <Num label="SMA50 k·ATR" value={params.sma50.k_atr} onChange={v=>setParams((p:any)=>({...p, sma50:{...p.sma50, k_atr:v}}))} />
            <Num label="Breakout lookback" step={1} value={params.breakout.lookback} onChange={v=>setParams((p:any)=>({...p, breakout:{...p.breakout, lookback:Math.max(20,Math.min(180,Math.round(v)))}}))} />
            <Num label="Breakout k·ATR" value={params.breakout.k_atr} onChange={v=>setParams((p:any)=>({...p, breakout:{...p.breakout, k_atr:v}}))} />
            <Num label="Stop ATR mult" value={params.stops.atr_mult} onChange={v=>setParams((p:any)=>({...p, stops:{...p.stops, atr_mult:v}}))} />
            <label className="flex items-center gap-2 text-sm">
              <span className="w-40 opacity-80">Stop MA</span>
              <select value={params.stops.ma} onChange={e=>setParams((p:any)=>({...p, stops:{...p.stops, ma:e.target.value}}))} className="w-28 rounded border bg-background px-2 py-1">
                <option value="sma50">SMA50</option>
                <option value="sma20">SMA20</option>
              </select>
            </label>
            <label className="flex items-center gap-2 text-sm">
              <span className="w-40 opacity-80">RSI min</span>
              <input type="number" value={params.filters.rsiRange[0]} onChange={e=>setParams((p:any)=>({...p, filters:{...p.filters, rsiRange:[Number(e.target.value), p.filters.rsiRange[1]]}}))} className="w-20 rounded border bg-background px-2 py-1"/>

            </label>
            <label className="flex items-center gap-2 text-sm">
              <span className="w-40 opacity-80">RSI max</span>
              <input type="number" value={params.filters.rsiRange[1]} onChange={e=>setParams((p:any)=>({...p, filters:{...p.filters, rsiRange:[p.filters.rsiRange[0], Number(e.target.value)]}}))} className="w-20 rounded border bg-background px-2 py-1"/>
            </label>
            <label className="flex items-center gap-2 text-sm col-span-2">
              <input type="checkbox" checked={params.filters.trendUp} onChange={e=>setParams((p:any)=>({...p, filters:{...p.filters, trendUp:e.target.checked}}))} />
              <span>Kræv positiv trend (SMA50&gt;SMA200 og pris&gt;200)</span>
            </label>
          </div>

          <div className="flex gap-2 pt-2">
            <button onClick={runBacktest} disabled={loading} className="rounded bg-primary text-primary-foreground px-3 py-1.5 text-sm disabled:opacity-60">
              {loading? 'Kører...' : 'Kør backtest'}
            </button>
            {error && <div className="text-sm text-red-500">{error}</div>}
          </div>
        </div>

        <div className="rounded-lg border p-4">
          <div className="text-sm font-medium mb-2">Equity (cumulativ R pr. trade)</div>
          <div className="h-64">
            {run?.equityCurve?.length ? (
              <LineChart labels={labels} data={eq} label="Equity (R)" color="#6b46c1" />
            ) : (
              <div className="h-full grid place-items-center text-sm opacity-60">Ingen data endnu</div>
            )}
          </div>
          <div className="pt-2 flex gap-2">
            <button disabled={!runId} onClick={()=>{ if(runId) window.open(`/api/backtest/csv?id=${encodeURIComponent(runId)}`, '_blank') }} className="rounded border px-3 py-1.5 text-sm disabled:opacity-50">Download trades (CSV)</button>
          </div>
        </div>
      </div>

      {run && (
        <div className="grid md:grid-cols-3 gap-4">
          <div className="rounded border p-3"><div className="text-xs opacity-70">Trades</div><div className="text-xl font-semibold">{run.summary?.trades ?? 0}</div></div>
          <div className="rounded border p-3"><div className="text-xs opacity-70">Hit rate</div><div className="text-xl font-semibold">{Math.round((run.summary?.hit_rate ?? 0)*100)}%</div></div>
          <div className="rounded border p-3"><div className="text-xs opacity-70">Expectancy (R)</div><div className="text-xl font-semibold">{run.summary?.expectancy_R ?? 0}</div></div>
        </div>
      )}

      {run?.perTicker?.length ? (
        <div className="rounded-lg border p-4">
          <div className="text-sm font-medium mb-2">Per ticker</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left opacity-70">
                <tr><th className="py-1">Ticker</th><th>Trades</th><th>Hit</th><th>Exp (R)</th></tr>
              </thead>
              <tbody>
                {run.perTicker.map((r:any)=> (
                  <tr key={r.ticker} className="border-t">
                    <td className="py-1">{r.ticker}</td>
                    <td>{r.trades}</td>
                    <td>{Math.round((r.hit_rate||0)*100)}%</td>
                    <td>{r.expectancy_R ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ): null}


      {trades?.length ? (
        <div className="rounded-lg border p-4">
          <div className="text-sm font-medium mb-2">Trades (viser op til 200)</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left opacity-70">
                <tr>
                  <th className="py-1">Ticker</th><th>Entry</th><th>Exit</th><th>Pris ind</th><th>Pris ud</th><th>R</th><th>%</th><th>Bars</th><th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {trades.slice(0,200).map((t:any, i:number)=> (
                  <tr key={i} className="border-t">
                    <td className="py-1">{t.ticker}</td>
                    <td>{t.entry_date}</td>
                    <td>{t.exit_date}</td>
                    <td>{t.entry_price}</td>
                    <td>{t.exit_price}</td>
                    <td>{t.r}</td>
                    <td>{t.pnl_pct}</td>
                    <td>{t.bars_held}</td>
                    <td>{t.exit_reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      <div className="rounded-lg border p-4">
        <div className="text-sm font-medium mb-2">Seneste backtests</div>
        <div className="flex flex-wrap items-end gap-2 mb-2">
          <label className="text-sm">
            <div className="opacity-70 text-xs">Ticker</div>
            <input value={tickerFilter} onChange={e=>setTickerFilter(e.target.value)} placeholder="f.eks. NVO" className="rounded border bg-background px-2 py-1"/>
          </label>
          <label className="text-sm">
            <div className="opacity-70 text-xs">Fra</div>
            <input type="date" value={dateFrom} onChange={e=>setDateFrom(e.target.value)} className="rounded border bg-background px-2 py-1"/>
          </label>
          <label className="text-sm">
            <div className="opacity-70 text-xs">Til</div>
            <input type="date" value={dateTo} onChange={e=>setDateTo(e.target.value)} className="rounded border bg-background px-2 py-1"/>
          </label>
          <button onClick={()=>{ setHistoryPage(1); loadHistory() }} className="rounded border px-3 py-1.5 text-sm">Anvend</button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left opacity-70">
              <tr><th className="py-1">Dato</th><th>Tickers</th><th>Lookback</th><th>Trades</th><th>Hit</th><th>Exp (R)</th><th></th></tr>
            </thead>
            <tbody>
              {history.map(it => (
                <tr key={it.id} className="border-t">
                  <td className="py-1">{new Date(it.created_at).toLocaleString()}</td>
                  <td>{Array.isArray(it.tickers)? it.tickers.join(', '): ''}</td>
                  <td>{Math.round((it.lookback_days||0)/365)} år</td>
                  <td>{it.summary?.trades ?? '-'}</td>
                  <td>{it.summary?.hit_rate != null ? Math.round(it.summary.hit_rate*100)+'%' : '-'}</td>
                  <td>{it.summary?.expectancy_R ?? '-'}</td>
                  <td><button onClick={()=>loadRun(it.id)} className="text-primary underline">Load</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between mt-2 text-sm">
          <div>Side {historyPage} af {Math.max(1, Math.ceil((historyTotal||0) / historyPageSize))} ({historyTotal} runs)</div>
          <div className="flex gap-2">
            <button disabled={historyPage<=1} onClick={()=>setHistoryPage(p=>Math.max(1,p-1))} className="rounded border px-3 py-1.5 disabled:opacity-50">Forrige</button>
            <button disabled={historyPage >= Math.ceil((historyTotal||0)/historyPageSize)} onClick={()=>setHistoryPage(p=>p+1)} className="rounded border px-3 py-1.5 disabled:opacity-50">Næste</button>
          </div>
        </div>
      </div>

    </div>
  )
}

