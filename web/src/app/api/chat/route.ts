export const runtime = 'nodejs'

// Minimal server-side chat proxy with model fallback and Supabase data summarization
// Expects env: HF_API_TOKEN, HF_MODEL_ID_PRIMARY, HF_MODEL_ID_FALLBACK (optional), SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET

type ChatBody = {
  prompt: string
  tickers?: string[]
  lang?: 'da' | 'en'
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
  const p = (k: string, d?: any) => (d == null ? '—' : d)
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
  if (data.position_health) parts.push(`pos=${data.position_health}`)
  if (Array.isArray(data.buy_zones) && data.buy_zones.length) {
    const zones = data.buy_zones.slice(0, 2).map((z: any) => `${p('l', z.low)}-${p('h', z.high)}`).join(', ')
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

export async function POST(req: Request) {
  try {
    const { prompt, tickers = [], lang = 'da', sessionId }: ChatBody = await req.json()
    if (!prompt || !prompt.trim()) {
      return new Response(JSON.stringify({ error: 'Missing prompt' }), { status: 400, headers: { 'content-type': 'application/json' } })
    }
    const safeTickers = Array.isArray(tickers) ? tickers.slice(0, 10) : [] // cap number for token budget

    // Load and summarize data
    const summaries: string[] = []
    for (const t of safeTickers) {
      try {
        const json = await fetchTickerFromSupabase(t)
        summaries.push(summarizeTicker(t, json))
      } catch (e) {
        summaries.push(`Ticker: ${t} | (kunne ikke hente data)`) // continue
      }
    }

    const input = buildInput(lang, prompt, summaries)
    const primary = (env('HF_MODEL_ID_PRIMARY', false) as string) || (env('HF_MODEL_ID', false) as string) || 'TheFinAI/finma-7b-full'
    const fallback = (env('HF_MODEL_ID_FALLBACK', false) as string) || 'TheFinAI/finma-7b-nlp'

    let answer = ''
    let usedModel = primary
    try {
      answer = await callHF(input, primary)
    } catch (e) {
      usedModel = fallback
      answer = await callHF(input, fallback)
    }

    // Optional: log to Supabase (if tables exist). Best-effort; ignore failures.
    // NOTE: For proper logging, create tables chat_sessions and chat_messages as described in README.
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

