"use client"

import React from 'react'

type Meta = { tickers?: string[] }

type Lang = 'da' | 'en'

export default function ChatWidget() {
  const [open, setOpen] = React.useState(false)
  const [meta, setMeta] = React.useState<Meta | null>(null)
  const [loadingMeta, setLoadingMeta] = React.useState(false)
  const [selected, setSelected] = React.useState<string[]>([])
  const [prompt, setPrompt] = React.useState('')
  const [lang, setLang] = React.useState<Lang>('da')
  const [answer, setAnswer] = React.useState<string>('')
  const [sending, setSending] = React.useState(false)

  React.useEffect(() => {
    if (!open || meta) return
    setLoadingMeta(true)
    ;(async () => {
      const tryFetch = async (url: string) => {
        const r = await fetch(url)
        if (!r.ok) throw new Error(String(r.status))
        return (await r.json()) as Meta
      }
      try {
        let j = await tryFetch('/api/data/meta.json')
        if (!Array.isArray(j.tickers) || j.tickers.length === 0) {
          j = await tryFetch('/data/meta.json')
        }
        setMeta(j)
      } catch {
        try {
          const j = await tryFetch('/data/meta.json')
          setMeta(j)
        } catch {
          setMeta({ tickers: [] })
        }
      } finally {
        setLoadingMeta(false)
      }
    })()
  }, [open, meta])

  function toggleTicker(t: string) {
    setSelected(prev => prev.includes(t) ? prev.filter(x => x !== t) : [...prev, t])
  }

  async function send() {
    if (!prompt.trim()) return
    setSending(true)
    setAnswer('')
    try {
      const res = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, tickers: selected, lang }) })
      const j = await res.json()
      if (!res.ok) throw new Error(j?.error || 'Server error')
      setAnswer(j.text || '')
    } catch (e: any) {
      setAnswer(e?.message || 'Fejl under forespørgslen')
    } finally {
      setSending(false)
    }
  }

  return (
    <>
      <button
        onClick={() => setOpen(v => !v)}
        className="fixed bottom-4 right-4 z-50 rounded-full bg-blue-600 text-white px-4 py-3 shadow-lg hover:bg-blue-500 focus:outline-none"
        aria-label="Open chat"
      >
        {open ? (lang === 'da' ? 'Luk chat' : 'Close chat') : 'AI Chat'}
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-2 sm:p-4">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <div className="relative w-full sm:max-w-xl bg-background text-foreground rounded-md shadow-xl border p-3 sm:p-4 max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-lg font-semibold">FinMA Chat</h2>
              <select value={lang} onChange={e => setLang(e.target.value as Lang)} className="border rounded px-2 py-1 bg-background">
                <option value="da">Dansk</option>
                <option value="en">English</option>
              </select>
            </div>

            <div className="mt-3">
              <label className="text-sm font-medium">Vælg tickers (max 10)</label>
              <div className="mt-2 max-h-40 overflow-auto border rounded p-2">
                {loadingMeta && <div className="text-sm opacity-70">Henter tickers...</div>}
                {!loadingMeta && (meta?.tickers?.length ? (
                  <div className="grid grid-cols-2 gap-2">
                    {meta!.tickers!.map(t => (
                      <label key={t} className="flex items-center gap-2 text-sm">
                        <input type="checkbox" checked={selected.includes(t)} onChange={() => toggleTicker(t)} />
                        <span>{t}</span>
                      </label>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm opacity-70">Ingen tickers fundet</div>
                ))}
              </div>
            </div>

            <div className="mt-3">
              <textarea
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                placeholder={lang === 'da' ? 'Skriv dit spørgsmål...' : 'Type your question...'}
                className="w-full border rounded p-2 bg-background"
                rows={4}
              />
            </div>

            <div className="mt-3 flex items-center gap-2">
              <button onClick={send} disabled={sending || !prompt.trim()} className="rounded bg-blue-600 text-white px-3 py-2 disabled:opacity-50">
                {sending ? (lang === 'da' ? 'Sender...' : 'Sending...') : (lang === 'da' ? 'Send' : 'Send')}
              </button>
              <button onClick={() => { setAnswer(''); setPrompt('') }} className="rounded border px-3 py-2">
                {lang === 'da' ? 'Ryd' : 'Clear'}
              </button>
            </div>

            {answer && (
              <div className="mt-3 border rounded p-2 whitespace-pre-wrap text-sm">
                {answer}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}

