1. Vision

Byg et integreret “Investment Copilot” dashboard hvor du:

Scanner nye aktier på fundamentale, tekniske og sentiment-parametre.

Får købs- og salgsniveauer (entry zones, stop loss, profit targets) beregnet og forklaret.

Holder din portefølje under opsyn i realtid – med health-score, risikoanalyse og diversificering.

Gemmer al historik i skyen, så du kan træne ML-modeller på dine egne data.

Bruger AI-assistenter direkte i dashboardet (FinBERT til nyhedssentiment, GPT-5-mini til fundamental+teknisk analyse og kontekstforklaring).

Får notifikationer når en aktie i watchlist/portefølje går ind i køb/stop/target-zoner.

2. Funktionelle kerneområder
A. Screener & Analyse

Input: tickerliste eller søgning

Output: scorer (fundamental/technical/sentiment), entry zones, stops, profit targets, flags.

Drilldown per ticker (charts, nøgletal, nyhedsfeed, AI-forklaring).

B. Porteføljeovervågning

Automatisk hentning af dine beholdninger (manuel eller broker API).

Visning af:

Aktuel værdi, P/L

Health score pr. aktie

Samlet risikoprofil (eksponering pr. sektor, volatilitet, diversificering)

Notifikationer når noget ændrer status (fx brud på SMA200 eller stopzone).

C. AI-lag

FinBERT: nyhedssentiment.

GPT-5-mini: genererer human-læsbare analyser og forklaringer af score/resultater (fundamental/technical).

Mulighed for at stille “hvad hvis”-spørgsmål i en chat widget.

D. Data & Historik

Alle snapshots gemmes i Supabase Storage (Parquet/CSV) + Postgres index.

Partitioneret per ticker/dato → let til ML senere.

Export/download til Excel.

E. Automatik & Notifikationer

Cronjobs der kører analyse hver 15. minut.

Notifikationer (webpush/email) når:

Ny aktie opfylder købszone-kriterier.

En porteføljeaktie rammer stop eller profit-mål.

Fundamentale ændrer sig drastisk.

3. Datapunkter (obligatoriske)
Fundamentale (minimum)

MarketCap, FCF, EBITDA, Revenue growth, NetDebt/EBITDA, Gross margin, Insider ownership, P/E, EV/EBITDA, evt. guidance ændringer.

Afledte metrics: FCF yield, ND/EBITDA, PEG, undervurdering vs 5-års median.

Tekniske (minimum)

SMA20/50/200, RSI14, ATR, ATR%, volumen trend, 52w high/low distance, Relative Strength mod benchmark.

Entry-zoner: pullbacks, breakout-retests.

Stop-niveauer: SMA50 − ATR, eller brugerdefineret R-risk.

Sentiment

FinBERT score for seneste 7/30 dage.

Flow-intensitet (nyhedsvolumen), signalord (“contract award”, “guidance raise”).

Punkt-score 0–25.

4. Overordnet Arkitektur
[ Next.js Frontend (Vercel) ]
  |
  +-- [ /api/screener ] -> henter live data & scorer
  |
  +-- [ /api/ingest ] -> gemmer snapshots i Supabase (Postgres + Storage)
  |
  +-- [ /api/portfolio ] -> viser aktuel portefølje + health
  |
  +-- [ Chat Widget ]
         +-- FinBERT microservice (sentiment)
         +-- GPT-5-mini (fundamental + teknisk analyse)


Cache/lagring: Supabase Postgres til index + Storage til Parquet/CSV.

Automatik: Vercel Cron pinger /api/ingest for at opdatere alle tickers + portefølje hver 15. minut.

5. Roadmap
Fase 1: Grundlæggende screener

Implementer /api/screener i Next.js (Node runtime).

Hent kursdata, beregn SMA/RSI/ATR, lav enkel fundamental scoring (MarketCap, FCF yield), og enkel FinBERT-score for nyheder.

UI: Leaderboard med ticker, pris, samlet score, RSI, ATR%, nyhedsantal.

Fase 2: Entry/Stop/Profit-zoner

Implementer heuristik for købszoner (pullbacks) og stop/target niveauer.

UI: Vis zoner grafisk i detaljevisning.

Notifikationer: enkel alert hvis aktie går ind i købszone.

Fase 3: Porteføljeovervågning

Tilføj porteføljeside (manuel indtastning af beholdning).

Health score pr. aktie (trend, stopdistance, sentiment).

Diversificeringsanalyse (fx sektorfaner).

Notifikationer ved stop/target.

Fase 4: Datahistorik og lagring

Byg /api/ingest der gemmer alle beregnede datapunkter i Supabase (snapshots + CSV/Parquet).

UI-knap “Eksporter” → download CSV.

Cronjobs hver 15. min.

Fase 5: AI-forklaringslag

Integrer GPT-5-mini i chat widget: tag dine data (score, zoner) og generér forklaring (“Hvorfor score 72?”).

FinBERT allerede inde, udbyg til flere signalord.

Fase 6: Risikostyring & avanceret analytics

Backtest parametre (hvor godt klarer dine entry/stop regler sig historisk).

Lav “what-if” scenarier i UI med GPT-5-mini.

Fase 7: ML-eksperimenter

Brug Supabase Storage som kilde til ML.

Start med simple modeller (klassificér “købszone ja/nej”).

Senere: regressionsmodel på dine features vs. fremtidig afkast.

6. Tekniske byggesten pr. lag
Lag	Anbefalet tech	Kommentar
Frontend	Next.js (App Router)	Tailwind, shadcn/ui, SWR for auto-refresh
API / Fetch	Next.js API Routes (Node)	yahoo-finance2, rss-parser, vader-sentiment
AI (sentiment)	FinBERT microservice	Kan hostes serverless
AI (analyse)	GPT-5-mini API	Kald med dine data som prompt
Historik/lagring	Supabase Postgres + Storage	CSV/Parquet partitioneret
Cron / Alerts	Vercel Cron + WebPush/Email	Forvarm cache + notifikationer
7. Minimal vedligehold

Alt i én kodebase (Next.js + API).

Supabase håndterer både objektlager og Postgres.

Vercel Cron driver alt.

FinBERT + GPT-5-mini kører som services du blot kalder; du vedligeholder ikke modellerne selv.

Schema versionering i din egen repo, så du kan udvide felter uden migrationshelvede.

8. Målbillede (når færdigt)

Du åbner dashboardet → ser live opdaterede scorer, entry/stop/target for alle tickers.

Din porteføljeside viser aktuel sundhed og risiko.

Notifikationer når en aktie går i køb/stop/target.

Du kan hente alle dine snapshots som CSV/Parquet med ét klik og træne ML-modeller på dem.

Chat-widgeten kan forklare “hvorfor” og lave what-if analyser.