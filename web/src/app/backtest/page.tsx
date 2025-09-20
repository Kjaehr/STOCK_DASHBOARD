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
  const [history, setHistory] = React.useState<any[]>([])

  async function loadHistory(){
    try{
      const r = await fetch('/api/backtest/list?limit=20', { cache:'no-store' })
      const j = await r.json()
      if(r.ok) setHistory(j?.items||[])
    }catch{}
  }
  React.useEffect(()=>{ loadHistory() }, [])

  async function loadRun(id:string){
    setLoading(true); setError(undefined)
    try{
      const r = await fetch(`/api/backtest?id=${encodeURIComponent(id)}`, { cache:'no-store' })
      const j = await r.json()
      if(!r.ok || j?.error) throw new Error(j?.error || `HTTP ${r.status}`)
      // Reconstruct UI summary shape from stored run
      const perTicker = Array.isArray(j?.run?.summary?.perTicker) ? j.run.summary.perTicker : []
      const summary = j?.run?.summary || {}
      const equityCurve = j?.run?.equity || []
      setRun({ summary, perTicker, equityCurve })
      setRunId(j?.run?.id)
    }catch(e:any){ setError(e?.message || String(e)) }
    finally{ setLoading(false) }
  }

  async function runBacktest(){
    setLoading(true); setError(undefined); setRun(null); setRunId(undefined)
    try{
      const body = { tickers: tickers.split(',').map(t=>t.trim()).filter(Boolean), lookbackYears, params }
      const r = await fetch('/api/backtest', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body) })
      const j = await r.json()
      if(!r.ok || j?.error){ throw new Error(j?.error || `HTTP ${r.status}`) }
      setRun(j); setRunId(j?.run_id); loadHistory()
    }catch(e:any){ setError(e?.message || String(e)) }
    finally{ setLoading(false) }
  }

  const labels = (run?.equityCurve||[]).map((p:any)=>String(p.i))
  const eq = (run?.equityCurve||[]).map((p:any)=>p.eq)

  return (
    <div className="p-4 md:p-6 max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Backtest</h1>
        <p className="text-sm opacity-80">Kør backtests over 1–5 år og justér parametre. Resultat gemmes i Supabase.</p>
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

      <div className="rounded-lg border p-4">
        <div className="text-sm font-medium mb-2">Seneste backtests</div>
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
      </div>

    </div>
  )
}

