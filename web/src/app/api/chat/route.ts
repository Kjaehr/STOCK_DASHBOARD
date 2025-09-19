export const runtime = 'nodejs'

// Server-side chat proxy with selectable engines and Supabase data summarization
// Engines:
// - model: 'finbert'  -> Hugging Face text-classification (ProsusAI/finbert) on recent news headlines
// - model: 'gpt5'     -> OpenAI Chat Completions (gpt-5-mini) for fundamental/technical analysis (includes FinBERT sentiment summary if available)
// Existing support (legacy): HF text-generation fallback list when no model specified
// Env expected:
//   HF_API_TOKEN (for HF calls), OPENAI_API_KEY (for gpt5),
//   HF_MODEL_ID_PRIMARY / HF_MODEL_ID_FALLBACK / HF_MODEL_LIST (legacy gen models),
//   HF_FINBERT_ID (default ProsusAI/finbert),
//   SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET

type ChatBody = {
  prompt: string
  tickers?: string[]
  lang?: 'da' | 'en'
  model?: 'finbert' | 'gpt5'
  sessionId?: string
}

function env(name: string, required = true): string | undefined {
  const v = process.env[name]
  if (!v && required) throw new Error(`Missing env: ${name}`)
  return v
}

async function fetchTickerFromSupabase(ticker: string) {
  const SUPABASE_URL = env('SUPABASE_URL') as string
  const SUPABASE_KEY = env('SUPABASE_KEY') as string
  const SUPABASE_BUCKET = env('SUPABASE_BUCKET') as string
  const url = `${SUPABASE_URL!.replace(/\/$/, '')}/storage/v1/object/${encodeURIComponent(SUPABASE_BUCKET!)}/${encodeURIComponent(ticker)}.json`
  const r = await fetch(url, { headers: { Authorization: `Bearer ${SUPABASE_KEY}` } })
  if (!r.ok) throw new Error(`Failed to fetch ${ticker}.json: ${r.status}`)
  return r.json()
}

function summarizeTicker(ticker: string, data: any): string {
  // Extract and compact key points; guard for missing fields
  const fmtPct = (v?: any) => (v == null ? '—' : `${Math.round(Number(v) * 1000) / 10}%`)
  const fmtNum = (v?: any) => (v == null ? '—' : String(v))
  const parts: string[] = []
  parts.push(`Ticker: ${ticker}`)
  if (data.score != null) parts.push(`score=${data.score}`)
  if (data.fund_points != null) parts.push(`fund=${data.fund_points}`)
  if (data.tech_points != null) parts.push(`tech=${data.tech_points}`)
  if (data.sent_points != null) parts.push(`sent=${data.sent_points}`)
  if (data.price != null) parts.push(`price=${data.price}`)
  if (data.sma50 != null) parts.push(`sma50=${data.sma50}`)
  if (data.sma200 != null) parts.push(`sma200=${data.sma200}`)
  if (data.rs_ratio != null) parts.push(`rs=${data.rs_ratio}`)
  // Add key fundamentals when available
  const f = data.fundamentals || {}
  if (f.fcf_yield != null) parts.push(`fcf_yield=${fmtPct(f.fcf_yield)}`)
  if (f.nd_to_ebitda != null) parts.push(`nd/ebitda=${fmtNum(f.nd_to_ebitda)}`)
  if (f.gross_margin != null) parts.push(`gm=${fmtPct(f.gross_margin)}`)
  if (f.revenue_growth != null) parts.push(`rev_g=${fmtPct(f.revenue_growth)}`)
  if (f.insider_own != null) parts.push(`insider=${fmtPct(f.insider_own)}`)
  if (f.pe != null) parts.push(`pe=${fmtNum(f.pe)}`)
  if (f.fwd_pe != null) parts.push(`fpe=${fmtNum(f.fwd_pe)}`)
  if (f.peg != null) parts.push(`peg=${fmtNum(f.peg)}`)
  if (f.p_s != null) parts.push(`ps=${fmtNum(f.p_s)}`)
  if (f.p_b != null) parts.push(`pb=${fmtNum(f.p_b)}`)
  if (f.ev_to_ebitda != null) parts.push(`ev/ebitda=${fmtNum(f.ev_to_ebitda)}`)
  // Useful technicals
  const t = data.technicals || {}
  if (t.rsi != null) parts.push(`rsi=${fmtNum(t.rsi)}`)
  if (t.atr_pct != null) parts.push(`atr%=${fmtNum(t.atr_pct)}`)
  if (data.position_health) parts.push(`pos=${data.position_health}`)
  if (Array.isArray(data.buy_zones) && data.buy_zones.length) {
    const zones = data.buy_zones.slice(0, 2).map((z: any) => `${fmtNum(z.low)}-${fmtNum(z.high)}`).join(', ')
    parts.push(`buy_zones=${zones}`)
  }
  return parts.join(' | ')
}

function systemPrompt(lang: 'da' | 'en') {
  if (lang === 'da') return (
    'Du er en finansiel assistent. Svar præcist og kortfattet.' +
    ' Brug KUN de leverede datauddrag som kilder. Angiv altid antagelser tydeligt.'
  )
  return (
    'You are a financial assistant. Respond precisely and concisely.' +
    ' Use ONLY the provided data excerpts as sources. State assumptions explicitly.'
  )
}

function buildInput(lang: 'da'|'en', prompt: string, summaries: string[]) {
  const intro = lang === 'da' ? 'Datauddrag (kompakte):' : 'Data excerpts (compact):'
  const question = lang === 'da' ? 'Spørgsmål:' : 'Question:'
  return `${systemPrompt(lang)}\n\n${intro}\n- ${summaries.join('\n- ')}\n\n${question}\n${prompt}`
}

function extractTextFromHF(json: any): string {
  // HF text-generation responses can vary; handle common shapes
  if (!json) return ''
  if (Array.isArray(json) && json.length) {
    const first = json[0]
    if (typeof first?.generated_text === 'string') return first.generated_text
    if (typeof first?.summary_text === 'string') return first.summary_text
  }
  if (typeof json?.generated_text === 'string') return json.generated_text
  if (typeof json === 'string') return json
  try { return JSON.stringify(json) } catch { return '' }
}

async function callHF(inputs: string, modelId: string, signal?: AbortSignal) {
  const token = env('HF_API_TOKEN') as string
  const r = await fetch(`https://api-inference.huggingface.co/models/${modelId}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ inputs, parameters: { max_new_tokens: 512, temperature: 0.3 } }),
    signal,
  })
  const text = await r.text()
  let json: any
  try { json = JSON.parse(text) } catch { json = text }
  if (!r.ok) {
    const msg = typeof json === 'string' ? json : JSON.stringify(json)
    throw new Error(`HF ${modelId} error ${r.status}: ${msg}`)
  }
  return extractTextFromHF(json)
}

// Small helpers
function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error('timeout')), ms)
    p.then(v => { clearTimeout(to); resolve(v) })
     .catch(e => { clearTimeout(to); reject(e) })
  })
}

async function fetchRssWithTimeout(url: string, ms = 4000): Promise<string | null> {
  const ac = new AbortController()
  const to = setTimeout(() => ac.abort('timeout'), ms)
  try {
    const r = await fetch(url, { signal: ac.signal })
    if (!r.ok) return null
    return await r.text()
  } catch {
    return null
  } finally {
    clearTimeout(to)
  }
}

// --- News fetching & FinBERT sentiment ---
function googleNewsUrl(query: string, lang: 'da'|'en') {
  const base = 'https://news.google.com/rss/search'
  const hl = lang === 'da' ? 'da' : 'en-US'
  const gl = lang === 'da' ? 'DK' : 'US'
  const ceid = lang === 'da' ? 'DK:da' : 'US:en'
  const q = encodeURIComponent(query)
  return `${base}?q=${q}&hl=${hl}&gl=${gl}&ceid=${ceid}`
}

function parseRssItems(xml: string): { title: string, link?: string }[] {
  const items: { title: string, link?: string }[] = []
  const itemRe = /<item[\s\S]*?<\/item>/g
  const titleRe = /<title><!\[CDATA\[(.*?)\]\]><\/title>|<title>(.*?)<\/title>/
  const linkRe = /<link>(.*?)<\/link>/
  for (const m of xml.matchAll(itemRe)) {
    const block = m[0]
    const t = block.match(titleRe)
    const l = block.match(linkRe)
    const title = (t?.[1] || t?.[2] || '').trim()
    const link = (l?.[1] || '').trim()
    if (title) items.push({ title, link })
  }
  return items
}

const NAME_MAP: Record<string,string> = {
  AAPL: 'Apple', MSFT: 'Microsoft', TSLA: 'Tesla', SPY: 'SPDR S&P 500',
  NVO: 'Novo Nordisk', 'NOVO-B.CO': 'Novo Nordisk',
}

async function fetchNewsHeadlines(tickers: string[], lang: 'da'|'en', perTicker = 8) {
  const headlines: string[] = []
  for (const t of tickers) {
    const name = NAME_MAP[t]
    const baseTerms = lang === 'da' ? ['aktie','nyheder'] : ['stock','news']
    const terms = Array.from(new Set([
      t,
      ...(name ? [name] : []),
      ...baseTerms.map(w => `${t} ${w}`),
      ...(name ? baseTerms.map(w => `${name} ${w}`) : []),
    ]))

    let collected = 0
    for (const q of terms) {
      if (collected >= perTicker) break
      const url = googleNewsUrl(q, lang)
      const xml = await fetchRssWithTimeout(url, 4000)
      if (xml) {
        const items = parseRssItems(xml)
        for (const it of items) {
          if (it.title && headlines.length < 100) {
            const before = headlines.length
            headlines.push(it.title)
            if (headlines.length > before) collected++
            if (collected >= perTicker) break
          }
        }
      }
    }
  }
  // de-dup across all
  return Array.from(new Set(headlines)).slice(0, 40)
}

async function classifyWithFinBert(texts: string[]) {
  if (!texts.length) return { counts: { POSITIVE: 0, NEGATIVE: 0, NEUTRAL: 0 }, details: [] as any[] }
  const token = env('HF_API_TOKEN') as string
  const modelId = (env('HF_FINBERT_ID', false) as string) || 'ProsusAI/finbert'
  const r = await fetch(`https://api-inference.huggingface.co/models/${modelId}`, {
    method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: texts })
  })
  const text = await r.text()
  let json: any
  try { json = JSON.parse(text) } catch { json = text }
  if (!r.ok) {
    const msg = typeof json === 'string' ? json : JSON.stringify(json)
    throw new Error(`HF ${modelId} error ${r.status}: ${msg}`)
  }
  // Expected: Array of arrays of {label,score}
  const counts = { POSITIVE: 0, NEGATIVE: 0, NEUTRAL: 0 } as Record<string, number>
  const details: { text: string, label: string, score: number }[] = []
  if (Array.isArray(json)) {
    json.forEach((arr: any, i: number) => {
      const best = Array.isArray(arr) ? arr.reduce((a: any, b: any) => (a?.score > b?.score ? a : b), null) : null
      if (best?.label) {
        const raw = String(best.label || '').toLowerCase()
        const cat = raw === 'positive' ? 'POSITIVE' : raw === 'negative' ? 'NEGATIVE' : 'NEUTRAL'
        counts[cat] = (counts[cat] || 0) + 1
        details.push({ text: texts[i], label: raw, score: best.score })
      }
    })
  }
  return { counts, details }
}

// --- OpenAI GPT-5 mini ---
function messagesToText(messages: {role:string, content:string}[]) {
  return messages.map(m => (m.role === 'system' ? `[system]\n${m.content}` : `[user]\n${m.content}`)).join('\n\n')
}

function extractOpenAIText(j: any): string {
  // Responses API common shapes
  if (typeof j?.output_text === 'string' && j.output_text.trim()) return j.output_text
  if (Array.isArray(j?.content)) {
    const parts = j.content.map((p: any) => p?.text?.value || p?.text || '').filter(Boolean)
    if (parts.length) return parts.join('\n')
  }
  if (Array.isArray(j?.output)) {
    const texts: string[] = []
    for (const item of j.output) {
      if (item?.type === 'message' && Array.isArray(item?.content)) {
        for (const c of item.content) {
          if (typeof c?.text?.value === 'string') texts.push(c.text.value)
          else if (typeof c?.text === 'string') texts.push(c.text)
        }
      } else if (item?.type === 'output_text' && typeof item?.text === 'string') {
        texts.push(item.text)
      }
    }
    if (texts.length) return texts.join('\n')
  }
  // Chat Completions API
  const cc = j?.choices?.[0]?.message?.content
  if (typeof cc === 'string' && cc.trim()) return cc
  if (Array.isArray(cc)) {
    const parts = cc.map((p: any) => p?.text || p?.value || '').filter(Boolean)
    if (parts.length) return parts.join('\n')
  }
  try { return JSON.stringify(j) } catch { return '' }
}

async function callOpenAI(messages: any[]) {
  const key = env('OPENAI_API_KEY') as string
  const ac = new AbortController()
  const timeout = setTimeout(() => ac.abort('timeout'), 30000)
  const input = messagesToText(messages)
  try {
    // Prefer Responses API for gpt-5 models
    let r = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'gpt-5-mini', input, max_output_tokens: 1800 }),
      signal: ac.signal,
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      const msg = err?.error?.message || `OpenAI error ${r.status}`
      // Retry with alternative param name if hinted
      if (/max_completion_tokens/i.test(String(msg))) {
        r = await fetch('https://api.openai.com/v1/responses', {
          method: 'POST',
          headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: 'gpt-5-mini', input, max_completion_tokens: 1800 }),
          signal: ac.signal,
        })
      } else if (/unknown_endpoint|use responses/i.test(String(msg))) {
        // Fallback to chat completions if responses not supported in env
        r = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: 'gpt-5-mini', messages }),
          signal: ac.signal,
        })
      }
    }
    const j = await r.json()
    if (!r.ok) throw new Error(j?.error?.message || `OpenAI error ${r.status}`)
    const text = extractOpenAIText(j)
    if (!text || !String(text).trim()) return '(Tomt svar fra GPT‑5 mini)'
    return text
  } catch (e: any) {
    if (e?.name === 'AbortError') throw new Error('OpenAI timeout (30s)')
    throw e
  } finally {
    clearTimeout(timeout)
  }
}

// Stream OpenAI Chat Completions as plain text chunks
async function streamOpenAIChat(messages: any[]) {
  const key = env('OPENAI_API_KEY') as string
  const ac = new AbortController()
  const timeout = setTimeout(() => ac.abort('timeout'), 60000)
  try {
    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'gpt-5-mini', messages, stream: true, temperature: 0.2 }),
      signal: ac.signal,
    })
    if (!r.ok || !r.body) {
      // Fallback: non-stream call, then stream once
      const full = await callOpenAI(messages)
      const enc = new TextEncoder()
      return new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(enc.encode(full))
          controller.close()
        }
      })
    }
    const reader = r.body.getReader()
    const dec = new TextDecoder()
    const enc = new TextEncoder()
    let buffer = ''
    let doneOuter = false
    return new ReadableStream<Uint8Array>({
      async pull(controller) {
        if (doneOuter) return controller.close()
        const { value, done } = await reader.read()
        if (done) {
          doneOuter = true
          controller.close()
          return
        }
        buffer += dec.decode(value, { stream: true })
        // Parse SSE lines
        let idx: number
        while ((idx = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, idx).trim()
          buffer = buffer.slice(idx + 1)
          if (!line) continue
          if (!line.startsWith('data:')) continue
          const data = line.slice(5).trim()
          if (data === '[DONE]') {
            doneOuter = true
            controller.close()
            return
          }
          try {
            const j = JSON.parse(data)
            const delta = j?.choices?.[0]?.delta?.content
            if (typeof delta === 'string' && delta) {
              controller.enqueue(enc.encode(delta))
            } else if (Array.isArray(delta)) {
              const txt = delta.map((p: any) => p?.text || p?.value || '').filter(Boolean).join('')
              if (txt) controller.enqueue(enc.encode(txt))
            }
          } catch {
            // Ignore JSON parse errors on keep-alives
          }
        }
      },
      cancel() {
        try { reader.cancel() } catch {}
      }
    })
  } finally {
    clearTimeout(timeout)
  }
}


export async function POST(req: Request) {
  try {
    const { prompt, tickers = [], lang = 'da', sessionId, model }: ChatBody = await req.json()
    if (!prompt || !prompt.trim()) {
      return new Response(JSON.stringify({ error: 'Missing prompt' }), { status: 400, headers: { 'content-type': 'application/json' } })
    }
    const safeTickers = Array.isArray(tickers) ? tickers.slice(0, 10) : [] // cap number for token budget

    // Load and summarize numeric data (used by gpt5 and legacy)
    const summaries: string[] = []
    for (const t of safeTickers) {
      try {
        const json = await fetchTickerFromSupabase(t)
        summaries.push(summarizeTicker(t, json))
      } catch (e) {
        summaries.push(`Ticker: ${t} | (kunne ikke hente data)`) // continue
      }
    }

    const url = new URL(req.url)
    const wantsStream = url.searchParams.get('stream') === '1' || ((req.headers.get('accept') || '').includes('text/plain') || (req.headers.get('accept') || '').includes('text/event-stream'))

    let answer = ''
    let usedModel = ''

    if (model === 'finbert') {
      // Sentiment on recent news headlines via FinBERT
      const headlines = await fetchNewsHeadlines(safeTickers, lang)
      if (!headlines.length) {
        const msg = lang === 'da' ? 'Ingen nyheder fundet for valgte tickers.' : 'No news found for selected tickers.'
        answer = `${msg}\n\nTickers: ${safeTickers.join(', ')}`
        if (wantsStream) {
          const enc = new TextEncoder()
          const stream = new ReadableStream<Uint8Array>({ start(controller){ controller.enqueue(enc.encode(answer)); controller.close() } })
          return new Response(stream, { status: 200, headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-cache, no-transform', 'x-model': 'finbert' } })
        }
        usedModel = (env('HF_FINBERT_ID', false) as string) || 'ProsusAI/finbert'
      } else {
        const res = await classifyWithFinBert(headlines)
        const counts = res.counts
        const header = lang === 'da' ? 'FinBERT sentiment (overskrifter)' : 'FinBERT sentiment (headlines)'
        const lines = [
          `${header}: POS=${counts.POSITIVE} | NEU=${counts.NEUTRAL} | NEG=${counts.NEGATIVE}`,
          '',
          ...(res.details.slice(0, 8).map(d => `- [${d.label}] ${d.text}`)),
        ]
        answer = lines.join('\n')
        if (wantsStream) {
          const enc = new TextEncoder()
          const stream = new ReadableStream<Uint8Array>({ start(controller){ controller.enqueue(enc.encode(answer)); controller.close() } })
          return new Response(stream, { status: 200, headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-cache, no-transform', 'x-model': 'finbert' } })
        }
        usedModel = (env('HF_FINBERT_ID', false) as string) || 'ProsusAI/finbert'
      }
    } else if (model === 'gpt5') {
      // Analysis via GPT-5 mini, include FinBERT sentiment summary if possible
      let sentimentLine = ''
      try {
        const headlines = await withTimeout(fetchNewsHeadlines(safeTickers, lang, 6), 5000)
        if (headlines.length) {
          const res = await withTimeout(classifyWithFinBert(headlines), 5000)
          sentimentLine = `FinBERT sentiment: POS=${res.counts.POSITIVE} | NEU=${res.counts.NEUTRAL} | NEG=${res.counts.NEGATIVE}`
        }
      } catch {}
      const system = lang === 'da'
        ? 'Du er en finansiel assistent. Lav kort, klar analyse baseret på tal (fundamental/teknisk).'
        : 'You are a financial assistant. Produce concise, clear analysis based on fundamentals/technicals.'
      const user = [
        summaries.length ? `Data:\n- ${summaries.join('\n- ')}` : '',
        sentimentLine,
        '',
        (lang === 'da' ? 'Spørgsmål:' : 'Question:'),
        prompt
      ].filter(Boolean).join('\n')
      if (wantsStream) {
        const stream = await streamOpenAIChat([
          { role: 'system', content: system },
          { role: 'user', content: user },
        ])
        return new Response(stream, { status: 200, headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-cache, no-transform', 'x-model': 'gpt-5-mini' } })
      }
      answer = await callOpenAI([
        { role: 'system', content: system },
        { role: 'user', content: user },
      ])
      usedModel = 'gpt-5-mini'
    } else {
      // Legacy: Hugging Face text-generation model(s) with fallback list (FinMA etc.)
      const input = buildInput(lang, prompt, summaries)
      const configuredList = ((env('HF_MODEL_LIST', false) as string) || '')
        .split(',').map(s => s.trim()).filter(Boolean)
      const primary = (env('HF_MODEL_ID_PRIMARY', false) as string) || (env('HF_MODEL_ID', false) as string)
      const fallback = (env('HF_MODEL_ID_FALLBACK', false) as string)
      const candidates = Array.from(new Set([
        ...configuredList,
        primary,
        fallback,
        'TheFinAI/finma-7b-full',
        'ChanceFocus/finma-7b-full',
        'TheFinAI/finma-7b-nlp',
        'ChanceFocus/finma-7b-nlp',
        'TheFinAI/FinMA-7B-NLP',
        'ChanceFocus/FinMA-7B-NLP',
      ].filter(Boolean)))
      const errors: string[] = []
      for (const m of candidates) {
        try { answer = await callHF(input, m); usedModel = m; break } catch (e: any) { errors.push(`${m}: ${e?.message || 'error'}`) }
      }
      if (!usedModel) {
        return new Response(JSON.stringify({ error: 'All model candidates failed', candidates, errors }), { status: 502, headers: { 'content-type': 'application/json' } })
      }
    }

    // Optional: log to Supabase (if tables exist). Best-effort; ignore failures.
    ;(async () => {
      try {
        if (process.env.CHAT_LOGGING !== '1') return
        const SUPABASE_URL = env('SUPABASE_URL') as string
        const SERVICE = (env('SUPABASE_SERVICE_ROLE', false) || env('SUPABASE_KEY', false)) as string
        if (!SUPABASE_URL || !SERVICE) return
        const base = `${SUPABASE_URL.replace(/\/$/, '')}/rest/v1`
        let sid = sessionId
        if (!sid) {
          const rs = await fetch(`${base}/chat_sessions`, { method: 'POST', headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}`, 'Content-Type': 'application/json', Prefer: 'return=representation' }, body: JSON.stringify({ user_id: null }) })
          const j = await rs.json()
          if (Array.isArray(j) && j[0]?.id) sid = j[0].id
        }
        if (!sid) return
        await fetch(`${base}/chat_messages`, { method: 'POST', headers: { apikey: SERVICE, Authorization: `Bearer ${SERVICE}`, 'Content-Type': 'application/json' }, body: JSON.stringify([{ session_id: sid, role: 'user', content: prompt, tickers: safeTickers }, { session_id: sid, role: 'assistant', content: answer, tickers: safeTickers }]) })
      } catch {}
    })()

    return new Response(JSON.stringify({ text: answer, model: usedModel }), { status: 200, headers: { 'content-type': 'application/json' } })
  } catch (e: any) {
    return new Response(JSON.stringify({ error: e?.message || 'Server error' }), { status: 500, headers: { 'content-type': 'application/json' } })
  }
}

