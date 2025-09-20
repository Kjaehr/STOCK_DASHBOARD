"use client"

import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import TextareaAutosize from 'react-textarea-autosize'
import { MessageSquare, Send, X, Loader2 } from 'lucide-react'

type Meta = { tickers?: string[] }

type Lang = 'da' | 'en'
type ModelChoice = 'finbert' | 'gpt5'

export default function ChatWidget() {
  const [open, setOpen] = React.useState(false)
  const [meta, setMeta] = React.useState<Meta | null>(null)
  const [loadingMeta, setLoadingMeta] = React.useState(false)
  const [selected, setSelected] = React.useState<string[]>([])
  const [prompt, setPrompt] = React.useState('')
  const [lang, setLang] = React.useState<Lang>('da')
  const [model, setModel] = React.useState<ModelChoice>('gpt5')
  const [answer, setAnswer] = React.useState<string>('')
  const [sending, setSending] = React.useState(false)
  const [messages, setMessages] = React.useState<{ id: number; role: 'user' | 'assistant'; content: string }[]>([])

  // Auto-preselect ticker based on current route (/ticker/[id]) when opening
  React.useEffect(() => {
    if (!open) return
    try {
      const path = window.location?.pathname || ''
      const m = path.match(/^\/ticker\/([^\/]+)$/)
      if (m && m[1]) {
        const id = decodeURIComponent(m[1])
        setSelected(prev => (prev.includes(id) ? prev : [id, ...prev].slice(0, 10)))
      }
    } catch {}
  }, [open])

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
    if (!prompt.trim() || sending) return
    const userMsg = { id: Date.now(), role: 'user' as const, content: prompt.trim() }
    setMessages(prev => [...prev, userMsg])
    setPrompt('')
    setSending(true)
    try {
      // Prepare placeholder assistant message for streaming
      const assistantId = Date.now() + 1
      setMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '' }])

      const res = await fetch('/api/chat?stream=1', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'text/plain' },
        body: JSON.stringify({ prompt: userMsg.content, tickers: selected, lang, model })
      })
      if (!res.ok) {
        const j = await res.json().catch(() => ({} as any))
        throw new Error(j?.error || 'Server error')
      }
      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      let full = ''
      if (reader) {
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          const chunk = decoder.decode(value, { stream: true })
          full += chunk
          setMessages(prev => prev.map(m => (m.id === assistantId ? { ...m, content: (m.content + chunk) } : m)))
        }
      } else {
        const text = await res.text()
        full = text
        setMessages(prev => prev.map(m => (m.id === assistantId ? { ...m, content: text } : m)))
      }
      setAnswer(full)
    } catch (e: any) {
      const err = e?.message || 'Fejl under forespørgslen'
      setMessages(prev => [...prev, { id: Date.now() + 2, role: 'assistant', content: `⚠️ ${err}` }])
      setAnswer(err)
    } finally {
      setSending(false)
    }
  }

  return (
    <>
      <button
        onClick={() => setOpen(v => !v)}
        className="fixed bottom-4 right-4 z-50 h-12 w-12 rounded-full bg-gradient-to-br from-violet-600 to-cyan-500 text-white shadow-lg shadow-black/20 hover:brightness-110 focus:outline-none flex items-center justify-center"
        aria-label="Open chat"
        title={open ? (lang === 'da' ? 'Luk chat' : 'Close chat') : 'AI Chat'}
      >
        {open ? <X className="h-5 w-5" /> : <MessageSquare className="h-5 w-5" />}
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-2 sm:p-4">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setOpen(false)} />
          <div className="relative w-full sm:max-w-xl bg-background/80 backdrop-blur text-foreground rounded-2xl shadow-2xl border p-4 max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-lg font-semibold">AI Chat</h2>
              <div className="flex items-center gap-2">
                <label className="text-sm opacity-80">Model</label>
                <select value={model} onChange={e => setModel(e.target.value as ModelChoice)} className="border rounded px-2 py-1 bg-background">
                  <option value="gpt5">GPT‑5 mini (Analyse)</option>
                  <option value="finbert">FinBERT (Sentiment)</option>
                </select>
                <select value={lang} onChange={e => setLang(e.target.value as Lang)} className="border rounded px-2 py-1 bg-background">
                  <option value="da">Dansk</option>
                  <option value="en">English</option>
                </select>
              </div>
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
              {/* Preset prompts tied to selected tickers */}
              <PresetRow
                lang={lang}
                selected={selected}
                onUse={(text) => { setModel('gpt5'); setPrompt(text); setTimeout(() => { if (text) send() }, 0) }}
              />
              <TextareaAutosize
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
                }}
                minRows={2}
                maxRows={8}
                placeholder={model === 'finbert'
                  ? (lang === 'da' ? 'Spørg om nyhedssentiment, fx: "Hvordan er stemningen for AAPL?"' : 'Ask for news sentiment, e.g., "What is the sentiment for AAPL?"')
                  : (lang === 'da' ? 'Skriv dit spørgsmål om fundamental/teknisk analyse...' : 'Ask about fundamental/technical analysis...')}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring mt-2"
              />
            </div>

            <div className="mt-3 flex items-center gap-2">
              <button onClick={send} disabled={sending || !prompt.trim()} className="inline-flex items-center gap-2 rounded-md bg-violet-600 text-white px-3 py-2 disabled:opacity-50 hover:bg-violet-500">
                {sending ? (<><Loader2 className="h-4 w-4 animate-spin" />{lang === 'da' ? 'Sender...' : 'Sending...'}</>) : (<><Send className="h-4 w-4" />{lang === 'da' ? 'Send' : 'Send'}</>)}
              </button>
              <button onClick={() => { setMessages([]); setAnswer(''); setPrompt('') }} className="rounded-md border px-3 py-2 hover:bg-accent">
                {lang === 'da' ? 'Ryd' : 'Clear'}
              </button>
            </div>

            <div className="mt-3 space-y-3 max-h-56 overflow-auto pr-1">
              {messages.map(m => (
                <div key={m.id} className={`flex items-start gap-2 ${m.role === 'user' ? 'justify-end' : ''}`}>
                  {m.role === 'assistant' && <div className="mt-1 h-8 w-8 rounded-full bg-gradient-to-br from-violet-500 to-cyan-400 flex items-center justify-center text-white"><MessageSquare className="h-4 w-4" /></div>}
                  <div className={`max-w-[80%] rounded-2xl border px-3 py-2 text-sm shadow-sm ${m.role === 'user' ? 'bg-primary text-primary-foreground rounded-tr-sm' : 'bg-card/60 backdrop-blur'}`}>
                    <div className="prose prose-invert prose-sm max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {m.content}
                      </ReactMarkdown>


                    </div>
                  </div>
                </div>
              ))}
              {sending && (
                <div className="flex items-start gap-2 opacity-80">
                  <div className="mt-1 h-8 w-8 rounded-full bg-gradient-to-br from-violet-500 to-cyan-400 flex items-center justify-center text-white"><MessageSquare className="h-4 w-4" /></div>
                  <div className="rounded-2xl border bg-card/60 backdrop-blur px-3 py-2 text-sm">
                    <div className="typing-dots"><span>.</span><span>.</span><span>.</span></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  )
}



function PresetRow({ selected, lang, onUse }: { selected: string[]; lang: 'da'|'en'; onUse: (text: string)=>void }){
  const one = selected[0]
  const many = selected.slice(0,3)
  const label = (txtDa: string, txtEn: string) => (lang==='da' ? txtDa : txtEn)
  const btn = (title: string, text: string) => (
    <button onClick={() => onUse(text)} className="text-xs px-2 py-1 rounded-full border hover:bg-accent">
      {title}
    </button>
  )
  const presets: Array<[string, string]> = []
  if (one) {
    presets.push([
      label(`Forklar score for ${one}`, `Explain score for ${one}`),
      label(`Forklar den samlede score og vigtigste drivere for ${one}.`, `Explain the total score and key drivers for ${one}.`)
    ])
    presets.push([
      label(`Koeb/stop/target for ${one}`, `Buy/stop/target for ${one}`),
      label(`Gennemgaa koebszone, foreslaaet stop og foerste target for ${one}.`, `Review buy zone, suggested stop and first target for ${one}.`)
    ])
    presets.push([
      label(`Fundamentale noegletal`, `Fundamental highlights`),
      label(`Opsummer FCF yield, vaekst, margin og vaerdiansaettelse for ${one}.`, `Summarize FCF yield, growth, margins and valuation for ${one}.`)
    ])
    presets.push([
      label(`RSI/Trend`, `RSI/Trend`),
      label(`Hvordan ser RSI, SMA50/200 og trend ud for ${one}?`, `What are RSI, SMA50/200 and trend for ${one}?`)
    ])
  } else if (many.length) {
    const list = many.join(', ')
    presets.push([
      label(`Sammenlign ${list}`, `Compare ${list}`),
      label(`Sammenlign score og vigtigste forskelle mellem ${list}.`, `Compare scores and key differences between ${list}.`)
    ])
  }
  if (!presets.length) return null
  return (
    <div className="flex flex-wrap gap-2 mb-2">
      {presets.map(([title, text], i) => (
        <React.Fragment key={i}>{btn(title, text)}</React.Fragment>
      ))}
    </div>
  )
}
