Stock Screener Dashboard — README.md (free-tier, no-server)

Goal: A zero-cost web dashboard that auto-scores stocks on Fundamentals + Technicals + Sentiment.
Stack: GitHub (repo + Actions + Pages), Python (data build), Static Web (Next.js/Vite), free RSS + yfinance.
Workflow: A scheduled GitHub Action fetches data and writes JSON into /data/. The static site reads and renders those JSON files.

0) What you’ll build

/scripts/fetch_compute.py — Python: pulls quotes (Yahoo via yfinance), headlines (Google News RSS), computes indicators & a 0–100 score, writes /data/TICKER.json.

/data/ — Pure static JSON artifacts committed by the Action (auto-versioned).

/web/ — Minimal static app (Next.js or Vite+React) that reads /data/*.json and shows a leaderboard + detail view.

/.github/workflows/nightly.yml — Cron to run Python, then commit the refreshed JSON.

GitHub Pages — Serves the web build and /data files for free.

No servers, no databases, no paid APIs.

1) Scoring model (transparent)

Total score (0–100) = 40% Fundamentals + 35% Technicals + 25% Sentiment.

Fundamentals (0–40 pts)

FCF Yield (TTM): >8% (12), 4–8% (8), 0–4% (4), <0% (0)

Net Debt / EBITDA: <1× (8), 1–2× (5), 2–3× (2), >3× (0)

Gross Margin (level as proxy for trend): >45% (8), 30–45% (4), else (0)

Revenue Growth YoY: >15% (6), 5–15% (3), 0–5% (1), <0% (0)

Insider Ownership: ≥10% (6), 3–10% (3), <3% (0)

If a metric is missing (common for microcaps), skip it and rescale the rest; flag “low data quality”.

Technicals (0–35 pts)

Trend: Close>SMA200 (+8) and SMA50>SMA200 (+4)

Momentum (RSI 14): 60–70 (+10), 45–60 (+6), >80 (+6), else (0)

Volume confirmation: rising 20-day avg & price>20-day mean (+7), rising only (+3)

ATR quality (ATR% of price): sweet-spot 2–6% (+6), edges 1–8% (+4), else (+1)

Sentiment (0–25 pts)

Headline sentiment (VADER, 7-day mean): >0.2 (+15), 0–0.2 (+8), <0 (0)

News flow intensity (7d count vs 30d avg): high (+6), neutral (+3), low (0)

Signal terms in titles (e.g., “contract award”, “guidance raise”, “insider buying”): +4 if present

2) Roadmap (sane, small steps)
v0.1 — Skeleton (1–2 hrs)

 Create repo: stock-dashboard

 Add /scripts/fetch_compute.py + /scripts/tickers.txt

 Add minimal /web (Vite+React or Next.js)

 Commit a manual /data/TEST.json to wire the web UI

 Publish GitHub Pages (from /web/dist or /web/out)

v0.2 — Data pipeline (same day)

 Implement Python data build (quotes, indicators, RSS, score)

 Create /.github/workflows/nightly.yml (cron 2×/day + manual dispatch)

 Action commits refreshed /data/*.json back to main

 Web reads /data/meta.json and renders the leaderboard

v0.3 — UX polish

 Search/filter tickers, color-coded score, tooltips for metrics

 Detail panel (charts for price vs SMA50/200, RSI, ATR%)

 Flag badges (e.g., no_debt_ebitda, low_data)

v0.4 — Watchlist control

 /scripts/tickers.txt is the watchlist (10–50 tickers)

 Edit list, push to main, the Action picks it up

v0.5 — Reliability

 Retry logic + graceful fallbacks if Yahoo/RSS hiccups

 Cache last good values in JSON (don’t blank on bad days)

v1.0 — Optional free upgrades

v0.6 — Portfolio (client-side, no backend)

	- Add a Portfolio page to add holdings (ticker, quantity, avg cost)
	- Persist positions in browser localStorage; provide Import/Export JSON
	- Compute live PnL, total value, weights using latest /data prices
	- Merge screening scores into portfolio view (e.g., highlight high-score holdings)


 Cloudflare Pages instead of GH Pages (faster CDN)

 Cloudflare Worker as free CORS proxy for on-demand fetch

 Add alternative data (EDGAR RSS, company PR feeds)

3) Local dev workflow (Augment Code + VS Code)

Clone repo → open in VS Code.

Python env:

cd scripts
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install yfinance pandas feedparser vaderSentiment


Watchlist: edit scripts/tickers.txt (one ticker/line).

Run build:

python scripts/fetch_compute.py


This creates /data/*.json and /data/meta.json.

Web app:

cd web
# Vite example
npm create vite@latest . -- --template react
npm i
npm run dev


In your app, fetch /data/meta.json, then load each /data/TICKER.json.

Augment Code tip: start an “Implemented Feature” session for each roadmap step; let it scaffold components/tests; keep commits small and descriptive.

Portfolio (local-only):

	- Data lives in your browser localStorage under key "portfolio".
	- Clear with: localStorage.removeItem('portfolio').
	- Import/Export via a JSON file to move between devices.


4) GitHub Actions (free scheduler)

Create file: .github/workflows/nightly.yml

name: Nightly Data Build
on:
  schedule:
    - cron: "15 4,16 * * *"    # ~06:15 & 18:15 CET/CEST
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: write   # allow committing /data
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install yfinance pandas feedparser vaderSentiment
      - run: python scripts/fetch_compute.py
      - name: Commit & push data
        run: |
          git config user.name "auto-bot"
          git config user.email "auto@users.noreply.github.com"
          git add data/*.json || true
          git commit -m "auto: refresh data $(date -u +%FT%TZ)" || echo "No changes"
          git push


Note (optional, Polygon.io EOD data):
- Add repository secret POLYGON_API_KEY in GitHub → Settings → Secrets and variables → Actions.
- To enable Polygon in the nightly build, set env in the workflow job:

  env:
    USE_POLYGON: "1"
    POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}

If USE_POLYGON is not set or the key is missing, the pipeline falls back to yfinance automatically.

5) Minimal file tree
stock-dashboard/
├─ data/                      # AUTO-GENERATED (committed by Action)
│  ├─ meta.json
│  └─ <TICKER>.json
├─ scripts/
│  ├─ fetch_compute.py
│  └─ tickers.txt
├─ web/                       # Static site
│  ├─ src/
│  │  ├─ main.tsx / index.tsx
│  │  └─ components/Leaderboard.tsx
	│  │  └─ components/Portfolio.tsx

│  ├─ public/
│  └─ package.json
└─ .github/
   └─ workflows/nightly.yml

6) fetch_compute.py — key responsibilities (natural language)

Read scripts/tickers.txt.

For each ticker:

Pull 1y daily OHLCV via yfinance.

Compute SMA50/200, RSI(14), ATR(14), ATR%, vol20 trend.

Pull Google News RSS titles for "TICKER stock", compute VADER sentiment and 7-day flow.

Fetch fundamentals from Ticker.info (FCF, market cap, totalDebt, EBITDA, grossMargins, revenueGrowth, heldPercentInsiders).

Map to fundamental points with the thresholds above.

Map technicals and sentiment to points.

Combine into score (0–100) via the 40/35/25 weights.

Write a compact JSON with:

ticker, score, fund_points, tech_points, sent_points

fundamentals{...}, technicals{...}, sentiment{...}

price, sma50, sma200, updated_at

flags (e.g., missing metrics).

After loop, write /data/meta.json with generated_at + tickers.

Keep the script idempotent and resilient: try/except per ticker, skip empties, don’t crash the whole run.

7) Frontend — what to implement

Leaderboard page (/):

Fetch /data/meta.json, then parallel fetch each /data/<T>.json.

Sort by score desc.

Table columns: Ticker, Score (color), Price, Fund, Tech, Sent, Flags.

Color: ≥70 green, 50–69 orange, <50 red.

Detail drawer/page (/ticker/:id):

Show:

Last price, SMA50/200, RSI, ATR% (single-series charts with Chart.js).

Fundamental metrics (raw + points).

Sentiment mean & article count (link “View on Google News”).

Flags.

UX niceties:

Client-side cache (localStorage) to avoid re-fetch on navigation.

Graceful empty-state if /data missing for a ticker.


Portfolio page (/portfolio):

	- Add position: Ticker, Quantity, Avg Cost
	- List positions with current Price, Value, Gain/Loss (abs/%), Score
	- Edit/remove positions; quick-add from Leaderboard row
	- Persist to localStorage; Import/Export JSON or CSV
	- Link tickers to detail pages

8) Deployment (GitHub Pages, free)

Option A: Vite (static)

npm run build → outputs to /web/dist

In repo settings → Pages → set Source to /web/dist via GitHub Actions (or move dist to /docs and point Pages to /docs).

Option B: Next.js (static export)

next.config.js → output: 'export'

npm run build && npm run export → outputs /web/out

Point Pages to /web/out.

Pages serves /data/ as plain files. Your app fetches them at /data/....

9) Free-tier constraints & tips

yfinance scrapes; be gentle: 10–50 tickers max in cron.

RSS is free; keep titles per ticker ≤25.

If rate-limited, backoff and keep previous JSON (don’t wipe).

Avoid client-side live fetch from Yahoo (CORS + rate limits). The Action prebuild is safer.

10) Troubleshooting

Action pushes nothing → Likely “No changes”; check that data/*.json actually updated.

Pages 404 on /data/ → Ensure Pages uses the same branch/folder where /data lives.

Missing fundamentals → Microcaps often lack Ticker.info; rely more on technicals/sentiment and show flags.

Charts blank → Your build path may differ. Use absolute /data/... or correct base in Vite/Next config.

11) Security & legal

This is educational. Not investment advice.

Respect robots.txt; we use yfinance & RSS (public endpoints).

Don’t store secrets; everything is public.

Portfolio privacy: Portfolio data is stored locally in your browser (localStorage) unless you choose to export it. No personal data is uploaded by default.


If you add a proxy (Cloudflare Worker), still stay within fair use.

12) Next ideas (still free)

Add EDGAR 8-K/10-Q headlines via SEC RSS.

Add insider transactions via form-4 RSS parsers.

Add pattern filters (breakout above 52-week high with volume).

Per-ticker notes stored as a markdown file in repo.

Optional portfolio sync (free): Use GitHub Gist via OAuth or Supabase free-tier to sync holdings across devices.

13) Quick start commands (copy-paste)

Python build locally

cd scripts
python -m venv .venv && source .venv/bin/activate
python -m pip install yfinance pandas feedparser vaderSentiment
echo -e "AAPL\nMSFT\nNVDA" > tickers.txt
python fetch_compute.py


Web (Vite)

cd web
npm create vite@latest . -- --template react
npm i
npm run dev
# wire up fetch('/data/meta.json') in Leaderboard component


Enable Action & Pages

Commit .github/workflows/nightly.yml.

Push main.

Repo Settings → Pages → set build output folder (see section 8).


## 14) Projektstatus (pr. 17-09-2025)

- Overblik
  - Repo-struktur matcher planen: `scripts/`, `data/`, `web/` er til stede. [DONE]
  - Python datapipeline: `scripts/fetch_compute.py` findes, og der ligger mange filer i `/data/*.json` inkl. `meta.json` → pipeline kører. [DONE]
  - Web: Next.js-konfiguration (`web/next.config.ts`) og `web/out` eksisterer → statisk eksport er bygget. [DONE]
  - Watchlist: `scripts/tickers.txt` findes. [DONE]
  - GitHub Actions (cron): Kører automatisk og committer opdateret `/data` til main. [DONE]
  - GitHub Pages/CDN: Live og serverer siten samt `/data`. [DONE]
  - Leaderboard/Detalje/Portfolio UI: Implementeret (baseline), viser data for alle tickers. [DONE]
  - Reliability (retry/cache): Delvist på plads (client-cache/defensive fetch). Pipeline-fallback kan udbygges. [PARTIAL]

- Hurtig status pr. roadmap
  - v0.1 — Skeleton: [DONE]
  - v0.2 — Data pipeline + Actions/Pages: Kører og opdaterer automatisk. [DONE]
  - v0.3 — UX polish (baseline: søgning, farvekoder, detaljer, charts): [DONE]
  - v0.4 — Watchlist control: Delvist (Add/Remove via GitHub issue-link). [PARTIAL]
  - v0.5 — Reliability: Delvist (client-cache; pipeline fallback/retry TBD). [PARTIAL]
  - v0.6 — Portfolio (local-only): Implementeret (import/export, PnL). [DONE]

- Næste minimale skridt (anbefalet rækkefølge)
  1) UI polish: konsolider styling (fx Tailwind + shadcn/ui), `ScoreBadge`, sticky header, zebra-rows, Detail Drawer med små charts.
  2) Pipeline-reliabilitet: try/except per ticker, backoff/retries, og fallback til sidste gyldige JSON; tilføj `schema_version` + klare `flags`.
  3) Watchlist-automation: udvid flowet, så Issues/PR labels kan opdatere `scripts/tickers.txt` automatisk via Action.
  4) Tests/QA: små Jest/RTL tests på UI; smoke-test og `--dry-run` til Python, samt CI-lint/format.

## 15) Forbedringsmuligheder (teknisk)

- Python datapipeline
  - Robusthed pr. ticker: `try/except` omkring alle eksterne kald + fallback til sidste gyldige JSON for at undgå “blanke” outputs.
  - Let cache/ETag: gem sidste response timestamps for at undgå unødvendige fetches.
  - Skema og flags: tydelig `schema_version`, `generated_at`, `flags` og `data_quality` pr. ticker.
  - Struktur: del `fetch_compute.py` i moduler (`fetch_quotes.py`, `compute_indicators.py`, `fetch_rss.py`, `score.py`).

- CI/CD
  - Actions: retries/backoff og artefakt-upload for logs ved fejl.
  - Cron: begræns tickers (10–50) og spred kørselstider for at være “rate-limit friendly”.

- Frontend datahåndtering
  - Defensive fetch: tomme/korrupt JSON håndteres med venlige empty-states og “Last updated”.
  - Client cache: `localStorage`/IndexedDB for hurtigere navigation og offline-læsning.

## 16) UI – gør det mere lækkert (hurtig gevinst + konsistens)

- Designsystem og tema
  - Brug Tailwind CSS + et komponentbibliotek (shadcn/ui eller Chakra) for hurtigt, ensartet design.
  - Farver: score→farve med farveblind-sikre paletter (≥70: grøn, 50–69: amber, <50: rød). Brug bløde gradients og 1–2 niveaus skygger.
  - Typografi: Inter/Geist med klare hierarkier (H1–H3, overline til flags).

- Komponenter og mikrointeraktioner
  - ScoreBadge: kompakt chip/pille med farve + tooltip (viser fordeling: Fund/Tech/Sent).
  - Leaderboard: zebra-rows, sticky header, kolonneikoner, hover-tilstand, klik åbner Detail Drawer.
  - Detail Drawer: små sparklines (pris, SMA50/200), mini-kort for RSI/ATR%, badges for flags.
  - Skeleton loaders + Empty-states: undgå “hop” i layout; vis venlige beskeder ved manglende data.
  - Dark mode toggle: system-default + manuel override.

- Diagrammer og ikoner
  - Chart.js eller Recharts med bløde gridlines, subtile punktmarkører og ens afstande/margener.
  - Ikoner: Tabler Icons eller Heroicons for rene, letlæselige symboler (fx flag, info, sentiment).

- Tilgængelighed og oplevelse
  - Kontrast ≥ WCAG AA, fokus-styles, tastaturnavigation på tabelrækker og knapper.
  - Ydelse: kode-split på Detail-komponent, memoization på tabelrækker, debounce på søgning.

- Konkrete TODOs (UI)
  - [ ] Installer Tailwind + vælg komponentbibliotek (shadcn/ui eller Chakra).
  - [ ] Implementér `ScoreBadge` og farvekodet `ScoreCell`.
  - [ ] Style Leaderboard-tabel (sticky header, zebra, hover, klik→drawer).
  - [ ] Byg `DetailDrawer` med 3 små charts (pris+SMA, RSI, ATR%).
  - [ ] Tilføj `DarkModeToggle` og gem preferencer i `localStorage`.
  - [ ] Tom-/fejltilstande og skeletons for alle sider.

- Deploy/UI verifikation
  - Efter build/export: åben Pages-URL og valider at: 1) `/data/meta.json` kan hentes, 2) farver/kontraster er korrekte i både lys/mørk tilstand, 3) tabel er responsiv (mobil/desktop).


## 17) Forbedringer til aktievurdering og købszoner (praktiske forslag)

Formål: Hurtigt vurdere en potentiel aktie og få konkrete købszoner/prisområder, hvor det statistisk giver bedst mening at akkumulere.

A) Fundamentale forbedringer (bedre kvalitet/valuation signaler)
- Sektorspecifikke tærskler: Justér FCF-yield, marginer og vækstkrav pr. sektor (Tech vs. Consumer vs. Industrials) og pr. market-cap (mega/large/mid/small).
- Kvalitet og kapitalallokering:
  - ROIC/CROIC (hvis tilgængeligt), FCF-margin, netto-margin-trend (3–5Y), operating leverage (ΔEBIT/ΔRevenue).
  - Dilution: aktieantal 3–5Y (udvanding) og SBC% af omsætning.
  - Earnings quality: accruals (NI – CFO), CFO/NI-ratio, stabilitet i CFO.
- Balance: Net cash/EBITDA, Interest coverage, Current/Quick ratio (faldbak hvis data mangler).
- Valuation (simple gratis-proxies):
  - P/FCF, EV/EBIT(DA), P/E (normaliser med 3–5Y medianer når muligt).
  - PEG-approx (P/E vs. vækst), Rule of 40 for software (vækst% + FCF-margin%).
  - Reverse DCF-lite: “krævet CAGR for at retfærdiggøre prisen” → lavt krav = bedre.
- Stabilitet: Varians i marginer/vækst over 3–5Y (jo lavere varians, jo højere stabilitetspoint).

B) Tekniske forbedringer (regime og kontekst til køb)
- Trend-regime: Klassificér som Uptrend (Close>SMA200 & SMA50>SMA200), Basing/Sideways, Downtrend.
- Relativ styrke: Prisratio vs. relevant sektor-ETF (fx MSFT/SPY eller XLK) – stigende ratio = +point.
- 52-ugers position: Afstand til 52w high/low; aktier tæt på 52w high i uptrend får bonus; dybe drawdowns i downtrend straffes.
- Volumenmønstre: Breakout med volumen over 20/50-dages gennemsnit, eller “volume shelf” konsolidering.
- ATR-/Keltner-bånd: Brug ATR% dynamisk for at vurdere “kvalitet” af trend og definere pullback-zoner.
- Enkle mønster-signaler: Golden/Death cross (SMA50↔SMA200), Higher Highs/Lows-count, mean reversion opsætning i stærk uptrend.

C) Sentiment/nyheder (robusthed og signaler)
- Kilder: Google + Bing + evt. RSS fra udvalgte finansmedier; deduplikér overskrifter.
- NER/event-detektion (simpelt lexicon): markér “guidance raise/cut”, “contract award”, “M&A”, “product launch”, “downgrade/upgrade”.
- Flow normalisering: Justér for ticker-popularitet (rolling baseline) så spikes vægtes korrekt.
- Polaritet og usikkerhed: Udvid VADER med leksika for guidance/earnings (simpelt ordsæt), lav bucket “pos/neu/neg/uncertain”.

D) Købszoner (prisområder der foreslås i JSON)
- Trend-baserede zoner (når Uptrend):
  - Pullback til SMA20/50: Zone = [SMA20 − k·ATR, SMA20] eller [SMA50 − k·ATR, SMA50], typisk k∈[0.5,1.5].
  - Retest af breakout: Seneste modstand → støtte; definer zone ±x% eller ±y·ATR omkring det niveau.
- Range/basing: Identificér range-high/low de sidste N dage; køb nær range-low hvis volumen falder ind i bunden og RS forbedres.
- Valuation-baserede zoner:
  - “Fair value”-interval fra simple multiples (fx blanding af (historisk median EV/EBIT, P/FCF) → prisinterval), og en “margin of safety” (fx 15–30%).
  - Reverse-DCF check: vis prisområde, hvor krævet vækst ≤ valgt tærskel (fx 8–10%).
- JSON-felter (nye):
  - buy_zones: [ { type: "sma_pullback"|"breakout_retest"|"valuation", price_low, price_high, confidence, rationale } ]
  - fair_value: { low, base, high, method }
  - risk: { atr_pct, dd_52w, beta? } og quality: { roic?, stability? }

E) UI/UX (hurtig beslutningsstøtte)
- “Why this score?”-tooltip per sektion: Vis pointbidrag pr. del-metric.
- “Buy Box” på details: Tydelige grønne zoner på prisgrafen + chips med prisintervaller og forklaring.
- Slider/toggles: Vælg “konservativ/neutral/aggressiv” profil → påvirker margin-of-safety og k-parametre for ATR-zoner.
- Alerts/badges: “Price entered buy zone” (client-side) + “Breakout with volume” badge.
- Relative strength mini-chart (ticker vs. sektor-ETF) og 52w distance indicator.

F) Konfiguration og skalering (uden server)
- Filbaseret config i repo: `config/weights.json`, `config/sector_thresholds.json`, `config/buyzones.json` (k, lookbacks, MoS%).
- Flag og schema: `schema_version`, `flags`, `data_quality`, samt logging af entries og kilder i CI-output for sporbarhed.
- Fallback-strategi: Hvis enkelte felter mangler (mikrocap), vis “partial confidence” men hold købszoner når tekniske data findes.

G) Verifikation, kalibrering og (enkel) backtest
- Simpel regelsbaseret backtest på daglige data: Køb når pris krydser ind i en aktiv “buy zone” i Uptrend og sælg ved (a) +10–20% mål, (b) ud af Uptrend, (c) 2×ATR stop.
- Evaluer hit-rate, gennemsnitligt afkast, max drawdown pr. strategi; justér k-parametre og MoS for bedste trade-off.
- Rapportér kalibrering i README (kort) og gem parametre i `config/`.

H) Konkrete næste skridt (1–2 uger)
1) Tilføj JSON-felter og UI-“Buy Box”: implementér ATR/SMA pullback-zoner og breakout-retests; vis dem i graf og som chips.
2) Tilføj relative strength og 52w-distance i technicals + vis indikator i UI.
3) Indfør `config/weights.json` og `config/buyzones.json` med profiler (konservativ/neutral/aggressiv).
4) Udvid sentiment-lexicon og log “entries=<n>, source=Google|Bing” pr. ticker i CI, for bedre observability.
5) (Valuation, trin 1) P/FCF og EV/EBIT median vs. nu → simpelt fair value-interval + MoS → generér valuation-zone.
6) (Kalibrering) En mini-backtest pr. strategi (ATR-pullback, breakout-retest) for 10–20 tickers over 1–2 år, og tilret k/MoS.

Output/JSON (eksempel):

```
{
  "buy_zones": [
    { "type": "sma_pullback", "price_low": 385.0, "price_high": 395.0, "confidence": 0.7, "rationale": "Uptrend; pullback til SMA50 ± 1.0 ATR" },
    { "type": "breakout_retest", "price_low": 412.0, "price_high": 416.0, "confidence": 0.6, "rationale": "Retest af modstand fra 52w-high; volumen normaliseret" },
    { "type": "valuation", "price_low": 360.0, "price_high": 380.0, "confidence": 0.5, "rationale": "P/FCF & EV/EBIT ~ 5Y median − 20% MoS" }
  ],
  "fair_value": { "low": 370.0, "base": 395.0, "high": 430.0, "method": "median multiples" },
  "risk": { "atr_pct": 2.8, "dd_52w": 12.5 },
  "quality": { "stability": 0.8 }
}
```

Noter (gratis/fri-tier): Vi forbliver inden for yfinance + RSS. Hvor data mangler, bruger vi konservative defaults og tydelige flags. Ingen tredjeparts betalte APIs.




python scripts/fetch_compute.py — to build data artifacts
python scripts/fetch_compute.py --backtest — to run the new mini-backtest






Skal vi stadig bruge GitHub Actions?
Kort svar: Du behøver ikke bruge Actions til at serve data, men du skal stadig have en “producer”, der genererer/opdaterer JSON’erne hver 15. minut.

Du har i dag en Python‑pipeline (scripts/fetch_compute.py). Der er to veje:

Hurtigste og mindst risikable (anbefalet nu):
Behold Python som er, men lad GitHub Actions køre hver 15. minut og uploade output til Supabase Storage i stedet for at committe til repoet.
Frontend læser herefter via vores nye /api/data‑route. Ingen rebuilds.
Fordele: Ingen omskrivning af Python → mindst arbejde, stabilt, gratis. Ulempen: Actions er stadig “motoren”, men det er backend‑arbejde og ikke en del af frontenden.
“Ren” serverless backend uden Actions (kræver ekstra arbejde):
Portér Python‑logikken til TypeScript og kør den i:
Supabase Edge Function + Supabase Cron (hver 15. min), eller
Vercel Serverless/Edge Function + Vercel Cron (hver 15. min)
Funktionen henter data (Yahoo/RSS), beregner indikatorer og skriver JSON til Supabase Storage.
Fordele: Alt kører “inde i backend”. Ulempe: Omskrivning af hele pipelinen fra Python → TS.
I praksis starter jeg typisk med første løsning (Actions som data‑producer), så UI og arkitektur er decouplet nu, og vi kan senere portere til Edge Functions hvis ønsket.

15-min opdatering — konkret plan (anbefalet)
Sæt disse Vercel envs (Project Settings → Environment Variables):
SUPABASE_URL, SUPABASE_KEY (anon key er ok til public read; service role er kun nødvendig hvis bucketen var privat)
SUPABASE_BUCKET = stockdash
NEXT_PUBLIC_DATA_BASE = /api/data
DATA_TTL_SECONDS = 900
Opdatér din GitHub Action til at:
Køre scripts/fetch_compute.py
Uploade alle data/*.json + data/meta.json til Supabase Storage “stockdash” med REST (curl PUT)
Efter første upload læser frontenden via /api/data/meta.json osv. — uden rebuild.
Eksempel på upload‑step (curl) pr. fil:

Vi kan selvfølgelig loope alle filer (inkl. meta.json). Jeg kan tilføje en komplet workflow for dig, når du siger til.

Alternativ: Ægte backend (uden Actions)
Hvis du vil helt væk fra Actions nu, foreslår jeg Supabase Edge Function + Cron hvert 15. min:

edge function fetcher Yahoo/RSS, beregner SMA/RSI/ATR/score, skriver til Storage
vores /api/data proxy forbliver uændret
kræver omskrivning af Python til TypeScript (kan gøres trinvist: start med meta.json + et par tickers)
Svar på dit spørgsmål
Ja, “backend” bør stå for dataopdatering — derfor enten:
Actions som midlertidig “backend‑worker”, der uploader til Storage (nuværende anbefaling), eller
Edge/Serverless cron‑jobs, når vi har portet Python til TS.
Frontend kalder kun vores serverless API (/api/data) og er dermed decouplet fra builds.
[NEXT STEPS]
Bekræft: Skal vi bruge den hurtige løsning nu (Actions uploader → Storage hver 15 min)?
Hvis ja:
Sæt Vercel envs (som ovenfor).
Giv grønt lys, så tilføjer jeg en GitHub Action workflow, der kører hvert 15. minut og uploader alle JSON‑filer til “stockdash”.
Hvis du hellere vil have ren serverless backend nu:
Jeg skitserer en Supabase Edge Function (TypeScript) med cron hvert 15. min og plan for gradvis port af Python.
[IMPROVEMENTS]
Per‑fil TTL: kortere TTL for meta.json, længere for historik; kan sættes i route baseret på path.
Bedre observability: log upload‑resultater i Actions; i API‑routen log status/hit‑miss (uden at eksponere nøgler).
Fail‑safe: ved upload‑fejl, behold sidste gyldige JSON (vi proxy’er, så UI fejler ikke hårdt).
Senere: skift til Supabase Edge Function + Cron for en 100% serverless pipeline uden Actions.

## 18) ML model artifacts → Hugging Face Hub (CI)

- CI uploads the latest ensemble bundle to a private Hugging Face model repo under: `Kjaehr/stock_dashboard_models` → `ensembles/`.
- What is uploaded from `ml_out/` after each training run:
  - Exactly one `.zip` bundle (`ensemble_<version>_<type>_<label>_<ts>.zip`), the matching `.json` metadata copy, `latest.json`, and `oof.csv` (+ per‑fold CSVs if present).
- Authentication: set GitHub Actions secret `HF_API_TOKEN`. No tokens are ever printed.
- Configuration (env, with defaults):
  - `HF_HUB_REPO_ID` (default `Kjaehr/stock_dashboard_models`)
  - `HF_HUB_SUBDIR` (default `ensembles/`)
- Output pointer (`ml_out/latest.json`) is rewritten post‑upload to:
  - `{ repo_id, path_in_repo, version, uploaded_at }` and points to the uploaded `.zip` on the Hub.
- Safety/robustness:
  - Upload retries (×3 with backoff) per file;
  - Size‑aware upload: `< 2GB` → standard `hf upload` (git/LFS‑style); `≥ 2GB` → `huggingface-cli upload` (multi‑part to S3).
  - Binary bundles are never committed.
- Where to find files:
  - Repo: https://huggingface.co/Kjaehr/stock_dashboard_models
  - Files: `.../resolve/main/ensembles/<filename>.zip`

To disable uploads locally, do nothing (default off). To force‑enable locally: `UPLOAD_TO_HF=true HF_API_TOKEN=... python scripts/ml/train_model_ensemble.py ...` (not required; CI handles uploads).

### Bundle format (one-file distribution)
- `<bundle>.zip` contains:
  - `meta.json` (version, timestamp, model_type, label_type, features, classes, seeds, datahash)
  - `preproc.joblib` (imputer, scaler, label_encoder)
  - `model.joblib` (compressed ensemble object)
  - `thresholds.json` (avg thresholds if available)

This is drop‑in compatible with the existing API usage: you still read the metadata JSON as before; services can download and open the `.zip` when they need the model.

### Health check (service start)
A simple loader/validator can be used in FastAPI service startup to detect missing/corrupt bundles early:

<augment_code_snippet mode="EXCERPT">
````python
import zipfile, json

def bundle_health_ok(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            for req in ("meta.json", "preproc.joblib", "model.joblib"):
                z.getinfo(req)
            json.loads(z.read("meta.json").decode("utf-8"))
        return True
    except Exception:
        return False
````
</augment_code_snippet>
