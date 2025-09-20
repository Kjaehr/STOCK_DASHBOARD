"use client"
import { useEffect, useMemo, useState } from 'react'
import { DATA_BASE } from '../../base'

type Rule = { id: string; ticker: string; type: 'BUY_ZONE'|'STOP_TOUCH'|'TARGET_TOUCH'; enabled: boolean }
type Event = { ts: string; ticker: string; type: string; title: string }

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return (await r.json()) as T
}

export default function AlertsPage(){
  const [tickers, setTickers] = useState<string[]>([])
  const [rules, setRules] = useState<Rule[]>([])
  const [events, setEvents] = useState<Event[]>([])
  const [selTicker, setSelTicker] = useState<string>('')
  const [types, setTypes] = useState<Record<string, boolean>>({ BUY_ZONE: true, STOP_TOUCH: false, TARGET_TOUCH: false })
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    ;(async () => {
      try {
        const meta = await fetchJSON<any>(`${DATA_BASE.replace(/\/$/, '')}/meta.json`)
        setTickers(Array.isArray(meta?.tickers) ? meta.tickers : [])
      } catch {}
      try {
        const r = await fetchJSON<{rules: Rule[]}>(`/api/alerts/rules`, { cache: 'no-store' })
        setRules(r.rules || [])
      } catch {}
      try {
        const e = await fetchJSON<{events: Event[]}>(`/api/alerts/events`, { cache: 'no-store' })
        setEvents(e.events || [])
      } catch {}
    })()
  }, [])

  const addRules = async () => {
    if (!selTicker) return
    const newOnes: Rule[] = ['BUY_ZONE','STOP_TOUCH','TARGET_TOUCH'].filter(t => types[t]).map((t) => ({ id: Math.random().toString(36).slice(2), ticker: selTicker, type: t as any, enabled: true }))
    const merged = [...rules.filter(r => r.ticker !== selTicker || !types[r.type]), ...newOnes]
    setSaving(true)
    try {
      await fetchJSON(`/api/alerts/rules`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ rules: merged }) })
      setRules(merged)
    } finally { setSaving(false) }
  }

  const toggleRule = async (id: string) => {
    const next = rules.map(r => r.id === id ? { ...r, enabled: !r.enabled } : r)
    setSaving(true)
    try {
      await fetchJSON(`/api/alerts/rules`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ rules: next }) })
      setRules(next)
    } finally { setSaving(false) }
  }

  const removeRule = async (id: string) => {
    const next = rules.filter(r => r.id !== id)
    setSaving(true)
    try {
      await fetchJSON(`/api/alerts/rules`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ rules: next }) })
      setRules(next)
    } finally { setSaving(false) }
  }

  return (
    <div className="space-y-6">
      <div className="rounded-lg border bg-card/60 backdrop-blur p-6">
        <h1 className="text-xl font-semibold mb-2">Alerts</h1>
        <p className="text-sm text-muted-foreground mb-4">Opret simple regler og se seneste events. V1: userless (deles via Storage).</p>

        <div className="flex flex-wrap gap-2 items-center mb-3">
          <select className="border rounded px-2 py-1" value={selTicker} onChange={e=>setSelTicker(e.target.value)}>
            <option value="">Vælg ticker…</option>
            {tickers.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          {(['BUY_ZONE','STOP_TOUCH','TARGET_TOUCH'] as const).map(t => (
            <label key={t} className="text-sm flex items-center gap-1">
              <input type="checkbox" checked={!!types[t]} onChange={e=>setTypes(s=>({ ...s, [t]: e.target.checked }))} /> {t.replace('_',' ')}
            </label>
          ))}
          <button className="px-3 py-1 rounded bg-blue-600 text-white disabled:opacity-50" disabled={!selTicker || saving} onClick={addRules}>Gem regler</button>
        </div>

        <div className="overflow-auto">
          <table className="w-full text-sm">
            <thead><tr><th className="text-left">Ticker</th><th className="text-left">Type</th><th>Status</th><th></th></tr></thead>
            <tbody>
              {rules.map(r => (
                <tr key={r.id} className="border-t">
                  <td className="py-1">{r.ticker}</td>
                  <td className="py-1">{r.type}</td>
                  <td className="py-1">
                    <button className="px-2 py-0.5 rounded border" onClick={()=>toggleRule(r.id)}>{r.enabled ? 'Enabled' : 'Disabled'}</button>
                  </td>
                  <td className="py-1 text-right">
                    <button className="px-2 py-0.5 text-red-600" onClick={()=>removeRule(r.id)}>Remove</button>
                  </td>
                </tr>
              ))}
              {rules.length === 0 && <tr><td className="text-muted-foreground py-2" colSpan={4}>Ingen regler endnu.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-lg border bg-card/60 backdrop-blur p-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold">Seneste events</h2>
          <button className="px-3 py-1 rounded border" onClick={()=>fetchJSON<{events:Event[]}>(`/api/alerts/events`, { cache: 'no-store' }).then(r=>setEvents(r.events||[]))}>Refresh</button>
        </div>
        <ul className="space-y-1">
          {events.map((e,i) => (
            <li key={e.ts+e.ticker+e.type+i} className="text-sm border-b py-1">
              <span className="font-medium">{e.ticker}</span> — {e.type} — {e.title}
              <span className="text-muted-foreground ml-2">{new Date(e.ts).toLocaleString()}</span>
            </li>
          ))}
          {events.length === 0 && <li className="text-sm text-muted-foreground">Ingen events endnu.</li>}
        </ul>
      </div>
    </div>
  )
}
