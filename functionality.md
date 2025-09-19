# Funktionalitet og Arkitektur – Stock Screener Dashboard

Senest opdateret: 2025-09-19
Formål: Én samlet teknisk beskrivelse af, hvordan hele systemet virker – fra datakilder og beregninger til frontend og udrulning.

## Systemoverblik (ende‑til‑ende)
- Serverless arkitektur: Frontend hostes på Vercel; data ligger i Supabase Storage (public bucket). Ingen egen server/database.
- Python‑pipeline bygger datafiler (JSON) pr. ticker + meta.json.
- Web (Next.js) henter data via /api/data proxy (Edge Route) som læser fra Supabase Storage.
- GitHub Actions kører periodisk (cron) og opdaterer data (commit og/eller upload til Storage).

Mappestruktur (uddrag):
- scripts/ — Python pipeline (fetch_compute.py, tickers.txt)
- data/ — Output JSON (auto‑genereret og committed)
- config/ — Vægte og parametre for scoring/købszoner
- web/ — Next.js app (statisk eksport muligt)
- docs/ — UI roadmap mv.

## Datapipeline (scripts/fetch_compute.py)
Hovedkørsel: python scripts/fetch_compute.py
- Læser tickers fra scripts/tickers.txt (én pr. linje; # er kommentar)
- For hvert ticker:
  1) Henter kursdata + beregner tekniske indikatorer
  2) Henter nyhedsoverskrifter + beregner sentiment
  3) Henter fundamentale nøgletal
  4) Scorer hver sektion og beregner samlet score (0–100)
  5) Foreslår “Buy Zones” og simple exit‑niveauer
  6) Skriver data/<TICKER>.json
- Efter loop: Skriver data/meta.json med generated_at og liste over succesfulde tickers

### Kilder og afhængigheder
- Kurs/teknik: yfinance (Yahoo Finance). Valgfrit Polygon.io (EOD) som primær for US‑tickers, ellers fallback til yfinance
- Fundamentals: yfinance.Ticker.info (typiske felter: freeCashflow, marketCap, totalDebt, ebitda, margins, growth, insiders)
- Nyheder/sentiment: Google News RSS (fallback: Bing News RSS) + VADER sentiment
- Beregninger: pandas

Miljøvariabler (valgfrie):
- USE_POLYGON=1 for at aktivere Polygon (kun US tickers)
- POLYGON_API_KEY til Polygon.io
- Cert‑workaround for Windows/OneDrive håndteres automatisk i scriptet

### Tekniske indikatorer (compute_indicators)
- Historik: 2 år daglige OHLCV
- SMA50 og SMA200 (trend)
- RSI(14) (momentum)
- ATR(14) og ATR% (volatilitet)
- Volumetrend: 20‑dages gennemsnit, “rising” check
- Price>MA20 indikator
- Serie til charts: sidste ~120 punkter (dates, close, sma50, sma200, rsi, atr_pct)
- Relative strength mod SPY (ratio + trend, hvis muligt)
- Flags sættes ved fejl/mangler (fx poly_fallback, rsi_fail, atr_fail, no_price_data)

### Nyheder og sentiment (fetch_news_sentiment)
- RSS søgning: "<SYMBOL> stock" OR "<LABEL> stock"
- Tidsvinduer: 7 og 30 dage
- VADER compound‑gennemsnit (7d) → bucket og point
- Flow‑intensitet (7d vs. 30d baseline) → point
- Signalord i titler (contract award, guidance raise, insider buying) → point
- Robust mod manglende data; returnerer neutral payload ved fejl

### Fundamentals (fetch_fundamentals + score_fundamentals)
- Udtrækker og normaliserer centrale nøgletal fra yfinance.info
- Konstruerer afledte nøgletal:
  - fcf_yield = FCF / MarketCap
  - nd_to_ebitda = NetDebt / EBITDA
- Scoring (0–40) ud fra tærskler (jf. README):
  - FCF‑yield, Net debt/EBITDA, Gross margin, Revenue growth, Insider ownership
- Mangler nogle metrics? Skalerer de tilgængelige og flagger low_data

### Teknik og sentiment scoring
- score_technicals (0–35): trend, RSI‑bucket, volume‑konfirmation, ATR‑kvalitet
- score_sentiment (0–25): VADER‑middelværdi, flow‑intensitet, signalord

### Vægte og samlet score
- Default i config/weights.json:
  { fundamentals: 0.40, technicals: 0.35, sentiment: 0.25 }
- Summen normaliseres til 1.0; total = round(wF*fund + wT*tech + wS*sent)

### Buy Zones og Exit‑niveauer
- Kun når optrend (Close>SMA200 og SMA50>SMA200)
- Købszoner (kan konfigureres i config/buyzones.json):
  - SMA50 pullback: [SMA50 − k·ATR_abs, SMA50]
  - SMA20 pullback: [SMA20 − k·ATR_abs, SMA20]
  - Breakout‑retest: recent high minus [k_lo·ATR_abs, k_hi·ATR_abs]
- Exit levels (heuristik): Stop≈SMA50−1·ATR; Targets≈+1.5R / +2.5R
- Position health: entry_readiness, exit_risk, in_buy_zone, dist‑to‑stop/target

### Robusthed, retries og fallback
- run_with_retry på eksterne kald (eksponentiel backoff)
- Per‑ticker fejl påvirker ikke hele kørslen
- Hvis en ticker fejler:
  1) Brug sidste gemte payload og markér flags: ["stale_data"]
  2) Som sidste udvej: skriv minimal stub med flag build_fail

## Outputformat (JSON)
Hvert ticker (data/<TICKER>.json) indeholder bl.a.:
- ticker, score, fund_points, tech_points, sent_points
- fundamentals{ fcf_yield, nd_to_ebitda, gross_margin, revenue_growth, insider_own, ... }
- technicals{ provider, trend{...}, rsi, atr_pct, vol_confirm{...}, series{...}, rs_ratio, rs_rising }
- sentiment{ mean7, count7, count30, flow, signal_terms, buckets, source, entries }
- buy_zones: [ { type, ma/level, price_low, price_high, confidence, rationale } ]
- exit_levels: { stop_suggest, targets[], rationale }
- position_health: { entry_readiness, exit_risk, in_buy_zone, dist_to_stop_pct, dist_to_t1_pct }
- price, sma50, sma200, updated_at, flags[]

Meta (data/meta.json):
- { generated_at: ISO8601, tickers: ["AAPL", "MSFT", ...] }

## Frontend (web/, Next.js)
- Bygget som App Router projekt (src/app/*)
- Dataadgang:
  - Produktion (Vercel): NEXT_PUBLIC_DATA_BASE=/api/data → web/src/app/api/data/[...path]/route.ts proxy’er til Supabase Storage via env: SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, DATA_TTL_SECONDS
  - Lokal udvikling: ../data kopieres til public/data; DATA_BASE falder tilbage til `${BASE}/data`
- Sider/komponenter (uddrag):
  - / (Leaderboard): tabellarisk oversigt, farvekodede scores, flags, søg/sortering
  - /ticker/[id]: detaljer (pris vs SMA, RSI, ATR%, fundamentals, sentiment, buy zones)
  - /portfolio: klient‑side portefølje i localStorage (import/eksport)
  - Komponenter: Leaderboard.tsx, Charts.tsx, Portfolio.tsx, theme‑toggle
- Styling: Tailwind (+ evt. shadcn/ui), dark mode via next‑themes
- Dev/build scripts (web/package.json):
  - npm run dev: kopierer ../data → public/data og starter next dev
  - npm run build: bygger appen; postbuild kopierer data til public/data
  - npm run export: statisk build + kopiering af data til out/data
- next.config.ts:
  - basePath/assetPrefix via NEXT_PUBLIC_BASE_PATH (valgfrit)
  - output: 'export' kan aktiveres via NEXT_OUTPUT_EXPORT=1 (for statisk eksport)

## Drift og CI/CD
- Hosting: Vercel (Next.js). Deploys udløses af GitHub‑push til main; miljøvariablerne ligger i Vercel Project Settings.
- Data build: GitHub Actions (cron) kører Python og opdaterer data. Issue‑workflows (add/remove) uploader allerede JSON til Supabase Storage; Nightly kan udvides med samme upload‑step.
- Data‑adgang i prod: Frontend kalder /api/data → Supabase Storage med cache/TTL (Edge Runtime).
- Centrale env vars (Vercel): SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, DATA_TTL_SECONDS, NEXT_PUBLIC_DATA_BASE=/api/data, (valgfrit) NEXT_PUBLIC_BASE_PATH.

## Konfiguration (config/)
- weights.json — vægte for samlet score (fund/tech/sent)
- buyzones.json — parametre for buy‑zone beregning (k for ATR, retest‑koefficienter)
- Yderligere konfiguration kan tilføjes file‑baseret for at holde alt gratis og revisionssporbar

## Lokalt udviklingsflow
1) Python miljø (fra README):
   - cd scripts
   - python -m venv .venv && aktiver
   - pip install yfinance pandas feedparser vaderSentiment
   - Redigér tickers.txt og kør: python fetch_compute.py (skriver data/*.json)
2) Web:
   - cd web
   - npm i
   - npm run dev (læser data fra ../data via public/data)

## Ydelse, pålidelighed og begrænsninger (free tier)
- yfinance er scraping‑baseret: hold universe lille (10–50 tickers) for at undgå rate limits
- RSS: begræns antal overskrifter pr. ticker
- Defensive patterns: retries, fallback til cached/stub, flags for datakvalitet
- Ingen hemmeligheder gemmes i repoet; optional nøgler (Polygon) via CI‑secrets

## Sikkerhed og privatliv
- Ingen persondata sendes til servere; portfolio‑data lever kun i browserens localStorage
- Alt data i /data/ er offentligt tilgængeligt (det er meningen)

## Kendte udvidelser (anbefalinger)
- Flere tekniske signaler: relative strength chart/badges, 52w‑distance
- Valuation‑zone (P/FCF, EV/EBITDA vs. 5Y median) og fair_value interval i JSON
- Bedre observability i Actions (log per ticker + artefakter)
- Mini‑backtest (allerede implementeret i scriptet med --backtest) til kalibrering af buy‑zone parametre

## Fejlfinding (hurtigt)
- “No changes” i Actions: check at data rent faktisk ændrede sig
- 404 på /data/ i host: sikr at Pages/hosting peger på samme branch/folder
- Manglende fundamentals: almindeligt for mikrocap; UI viser flags, score skaleres

## Resume
- Producer: Python genererer strukturerede JSON‑artefakter, robust mod fejl
- Konsument: Next.js præsenterer transparente, forklarlige scorer
- Drift: Vercel + Supabase Storage + GitHub Actions (cron) → enkelt, billigt og vedligeholdeligt

