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
