# Roadmap – Investment Copilot (Vercel + Supabase)

Senest opdateret: 2025-09-19
Kildegrundlag: Vision.md og Complete_Goal.md
Formål: Klar, faseopdelt plan med leverancer og acceptance criteria til at bygge systemet.

## Samlet strategi
- Serverless: Next.js (Vercel) + Supabase (Storage + Postgres), Cron hvert 15. minut
- On-demand API: Live fetch (yahoo-finance2, RSS, VADER) i Next API-routes, med cache (revalidate=900) og manuel refresh (refresh=1)
- Data lake: Supabase Storage (Parquet/CSV) + Postgres-index til hurtige queries
- AI: FinBERT til nyhedssentiment; GPT‑5‑mini til forklaringer/“what‑if”
- Notifikationer: WebPush/Email på vigtige events (købszone, stop, targets)

---

## Fase 0 – Baseline og miljø (1–2 dage)
Mål: Ensret prod/dev og sikre runtime, env vars og basis-cache.
- Leverancer
  - Vercel env vars sat: SUPABASE_URL, SUPABASE_KEY (server), SUPABASE_BUCKET, DATA_TTL_SECONDS=900, NEXT_PUBLIC_DATA_BASE=/api/data
  - Runtime‑valg pr. route (Node hvor yahoo-finance2/rss-parser bruges)
  - Caching-strategi: next revalidate=900 + SWR i UI
- Acceptance criteria
  - API routes bygger og svarer under 2s for 1–5 tickers
  - Leaderboard viser Endpoint, tid til seneste update, manual “Refresh” virker
- Noter
  - Rate-limit venlighed: batch størrelse ≤10 per request; semafor på parallel fetch

## Fase 1 – Screener API + UI (fund/tech/sent) (3–5 dage)
Mål: /api/screener (Node) returnerer fulde metrics + score; UI viser resultater.
- Backend
  - Implementér /api/screener?tickers=A,B,…
  - Hent historik (2y, 1d) → beregn SMA20/50/200, RSI14, ATR/ATR%; volumen-trend; RS mod benchmark (valgfrit)
  - Fundamentals (quoteSummary): marketCap, ebitda, freeCashflow → fcf_yield, nd/ebitda (hvis muligt)
  - Nyheder: Google News RSS (fallback Bing) → VADER mean7, flow, signalord
  - Scoring: weights fra config/ (paritet til Python/README
  - Flags + defensive retries/backoff
- Frontend
  - Leaderboard: ticker, price, score, fund/tech/sent delscore, RSI, ATR%, flags
  - SWR med refreshInterval=15m, revalidateOnFocus=true; “Opdater”-knap (refresh=1)
- Acceptance criteria
  - Score = 0–100; delscore synlige i tooltip
  - Empty/loading/error states håndteres pænt
  - Safari/Chrome/Firefox parity

## Fase 2 – Entry/Stop/Targets & Buy Zones (2–4 dage)
Mål: Heuristik for entry/stop/targets og visning i Details.
- Backend
  - Buy zones: SMA50 ± k·ATR, SMA20 ± k·ATR, breakout-retest (konfigurerbar via config/buyzones.json)
  - Exit levels: stop≈SMA50−1·ATR; targets≈+1.5R/+2.5R
  - Position health: entry_readiness, exit_risk, in_buy_zone
- Frontend
  - Detailvisning med mini-charts (pris, SMA, RSI, ATR%); chips for buy zones/targets
- Acceptance criteria
  - Buy zone/exit vises deterministisk og forklares (tooltip)
  - In‑buy‑zone markering matcher backend

## Fase 3 – Portefølje & Health (2–4 dage)
Mål: Lokal portefølje med health score og diversificering.
- Frontend
  - Portfolio side: CRUD (ticker, qty, avg cost) i localStorage
  - Health badge pr. position (trend, in_buy_zone, dist til stop)
  - Diversificering (simpel sektor/allokering når data findes)
- Backlog (senere): Broker API integration (stub)
- Acceptance criteria
  - P/L og totals korrekt; ingen konsolfejl; data persisterer lokalt

## Fase 4 – Data Lake & Ingest (4–7 dage)
Mål: /api/ingest gemmer snapshots i Supabase Postgres + Storage (CSV/Parquet), klar til ML/eksport.
- Backend
  - /api/ingest?tickers=A,B: henter samme data som screener → upsert i Postgres.snapshots (idempotent), upsert CSV, skriv Parquet pr. kørsel (partitioneret path)
  - /api/download (signed URL til CSV pr. ticker), /api/query-to-csv (stream Postgres som CSV)
  - Manifests/latest.json for seneste batch
- Datamodeller
  - snapshots: timestamp_iso, ticker, pris/teknik, fundamentals, sentiment, score, flags
  - files: sti, størrelse, checksum, created_at
- Acceptance criteria
  - CSV append konsistent; Parquet filer læsbare (DuckDB lokalt)
  - Enkle Postgres queries til UI (fx seneste snapshot pr. ticker)

## Fase 5 – Cron & Cache Forvarmning (1–2 dage)
Mål: Stabil 15‑min opdatering uden rebuilds.
- Opsæt Vercel Cron:
  - /api/ingest i batches (fx 3–5 kald á 10–15 tickers)
  - Evt. /api/screener subset for varme caches
- Acceptance criteria
  - Data opdateres indenfor 15–20 min. SLA
  - Ingen timeouts; log rate-limit hændelser


> Note (Vercel Hobby limitation): Cron Jobs kører kun 1 gang pr. dag på Hobby-planen. Verden er midlertidigt sat til daglige kørsler i `web/vercel.json` (kl. 06:00–06:20 UTC i forskudte trin). Når du opgraderer til Pro, ændres udtrykkene tilbage til hvert 15. minut:
>
> - Ingest gruppe 0/1/2: `*/15 * * * *`, `2-59/15 * * * *`, `4-59/15 * * * *`
> - Warm: `1-59/15 * * * *`
> - Alerts: `6-59/15 * * * *`
>
> Alternativt kan alle sættes til `*/15 * * * *` og spredes i kode.

## Fase 6 – Alerts/Notifikationer (2–4 dage)
Mål: WebPush/Email ved køb/stop/target events.
- Backend
  - Simpel regel: “Notify when ticker enters buy zone” + “stop/target touch”
  - Persist regler i Postgres (userless v1: lokal/eksport; v2: simpel auth)
- Frontend
  - Alerts UI: create/enable/disable; historik over seneste events
- Acceptance criteria
  - Push/email leveres (test-kanal); duplikerede events de‑dupes per 24h

## Fase 7 – AI‑lag (2–5 dage)
Mål: Chat med FinBERT + GPT‑5‑mini forklaringer.
- Backend
  - /api/chat: allerede i repo; sikre summarization fra Supabase og HF/OpenAI integration
- Frontend
  - ChatWidget knyttet til valgte tickers; preset prompts (fx “Forklar score”)
- Acceptance criteria
  - Lav latenstid (<4s for korte svar), robust ved manglende data

## Fase 8 – Backtest & Param‑kalibrering (3–6 dage)
Mål: Evaluér buy‑zones/regler og justér parametre.
- Backend
  - Mini‑backtest over historik (10–20 tickers, 1–2 år) for SMA/ATR‑baserede setups
  - Rapporter: hit‑rate, avg win/loss, expectancy
- Frontend
  - Simpel rapportside (tabeller/grafer)
- Acceptance criteria
  - Parametre opdateres i config/ med versionsmærkning og effekt på scorer dokumenteres

## Fase 9 – ML‑eksperimenter (5–10 dage, fleksibel)
Mål: Første model(r) på egne data.
- Data
  - Parquet/CSV fra Storage; DuckDB/Notebook pipeline
- Modeller
  - Klassifikation: “købszone → efterfølgende +X%/−Y% inden N dage”
  - Evt. regressionsmodel for afkast over N dage
- Acceptance criteria
  - Baseline‑metrikker dokumenteret; reproducérbar notebook + exporteret features

## Fase 10 – Hardening, Sikkerhed & Observability (løbende)
Mål: Robust drift, lav vedligehold.
- Observability: structured logs, latency/error‑rate, rate‑limit counters
- Sikkerhed: brug server‑only keys, signed URLs, (senere) RLS med enkel auth
- Ydelse: concurrency‑limit, tidsbudgetter, cache‑hit‑rate
- QA: visuelle snapshots, røgsignal‑tests på nøgleflows
- Doc: functionality.md, Vision.md, ROADMAP.md holdes opdateret

---

## Afhængigheder og cut‑lines
- Node runtime kræves for yahoo-finance2 og rss-parser.
- Batchdisciplin og caching er nødvendige for at undgå rate‑limits/timeouts.
- Cut‑line efter Fase 2: Systemet er nyttigt uden historik/ML.
- Cut‑line efter Fase 5: Fuldt “low‑maintenance” loop (cron + download) på plads.

## Risici og mitigering
- Rate‑limits fra Yahoo → cache (revalidate=900), batch, retries/backoff, evt. fallback til Supabase‑cache/“last good”.
- Funktionstimeouts → mindre batches, parallel‑limit (fx 5), hurtigere timeouts pr. kald.
- Datakonsistens (CSV append) → skriv til tmp + atomic rename; daglig merge job hvis nødvendigt.
- Privacy/keys → kun server‑routes bruger service‑keys; client ser aldrig dem.

## Milepæle (anbefalet rækkefølge)
1) Fase 0–1 (Screener MVP live)
2) Fase 2 (Buy‑zones/Details)
3) Fase 3 (Portfolio health)
4) Fase 4–5 (Ingest + Cron + Download)
5) Fase 6–7 (Alerts + AI)
6) Fase 8–9 (Backtest + ML)

## Definition of Done (per fase)
- Acceptance criteria opfyldt
- Ingen konsolfejl; defensive states (Loading/Empty/Error)
- Ydelse: p50 < 2s API (små batches), p95 < 4s
- Dokumentation opdateret (funktionality.md + ROADMAP.md)

## Næste skridt (konkrete)
- A) Bekræft hybrid vs. fuld runtime‑API – vi starter med hybrid (anbefalet)
- B) Start Fase 0–1 branche: feat/api-screener-v1 (Node runtime, cache, SWR)
- C) Book Cron (hver 15. min) – begynd med 1 batch; mål latency og rate‑hits

