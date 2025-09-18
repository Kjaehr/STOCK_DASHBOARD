# UI Roadmap – Stock Dashboard (Dark + Modern Admin Kit)

Status: v1 (Foundation DONE; Phase 2 in progress)
Owner: UI/Frontend
Last updated: 2025-09-18

## Vision
Skabe et moderne, lækkert dashboard i dark mode med "glass"-look, komfort spacing og en stram, konsistent component‑stil. UI’et skal føles hurtigt, roligt og professionelt – og være let at vedligeholde.

## Designprincipper
- Klar hierarki: tydelig typografi, luft, og cards med subtile borders/shadows
- Konsistens: én spacing‑skala, én farvepalette, samme komponentmønstre
- Non‑blocking: tom‑/loading‑/error‑tilstande der ikke blinker for meget
- Hurtigt: små bundles, cache‑smart data, skeletons i stedet for spinners
- Transparens: badges for datakilde/endpoint og “last updated”

## Farver, tema og tokens
- Mode: Dark default (system tilladt)
- Accents: Violet (#7c3aed) og Turkis (#06b6d4) – kan bruges i gradienter
- Glass: bg-…/60 + backdrop-blur på topbar og cards i udvalgte steder
- Spacing scale: 4/8/12/16/24/32
- Brug eksisterende shadcn/tailwind tokens (bg-background, text-foreground, border, card, muted, etc.) – ingen nye dependencies uden accept

## IA (informationsarkitektur)
- Sidebar: Dashboard, Leaderboard, Portfolio, Alerts, Settings (+ kommende)
- Topbar: global søgning, handlinger (Refresh), theme toggle, links

---

## Faser og leverancer

Hver fase afsluttes med acceptance criteria, visuel QA og small e2e smoke.

### Fase 1 – Foundation & Shell (DONE)
- DashboardShell: sidebar + glass topbar + content container (max-w-7xl)
- Logo/ikon i venstre hjørne (violet→turkis gradient)
- Dark som default
- Placeholder‑sider for Alerts/Settings
Acceptance criteria
- [x] App kører med ny shell uden at bryde eksisterende sider
- [x] Sidebar-nav synlig på ≥md, topbar sticky med blur
- [x] Lighthouse Performance ≥ 90 (uden data fetching i test)

### Fase 2 – Leaderboard modernisering
UI
- Header med: titel, søgning, presets som chips/pills, Refresh, Endpoint‑badge
- 3–4 StatCards over tabellen (Tickers, Avg score, Updated, Polygon usage?)
- Tabel: sticky header, zebra rows, forbedret spacing, subtle hover
- Badges: score/fund/tech/sent tooltips; flags i compact overflow
- Loading: skeleton rows; Error: subtle alert; Empty: rolig tom‑tilstand
- Responsive: min‑width tabel + vandret scroll i card
Tech/UX
- URL‑synk for søgning/sort/preset (deep‑linking)
- Keyboard: / fokus i søgning; Enter = åbn Details
Acceptance criteria
- [ ] 95%+ CLS‑fri interaktion (ingen layout shift ved load)
- [ ] Keyboard shortcuts virker og er dokumenteret
- [ ] Ingen layout‑overflow på mobil (<360px) udover kontrolleret hor. scroll i tabel

### Fase 3 – Ticker Detail (cards + moderne chart)
UI
- 2–3 kolonne grid af cards: Price Chart (toolbar range), Technicals, Fundamentals, Sentiment, Buy Zones/Exits
- Badges: Source (polygon/yfinance), Endpoint, Updated
- Chips: buy‑box/exit med tydelig semantik og tooltips
- Mini RS vs SPY (lille sparkline/badge) – hvis data
Tech/UX
- Chart palette (muted gridlines, subtle legend) og "overlay only" styling
- Loading skeletons for hver card; error‑inline uden at bryde grid
Acceptance criteria
- [ ] Ingen blocking spinners; alt har skeleton/fallback
- [ ] Chart føles roligt (ingen blink ved data‑update)

### Fase 4 – Portfolio (revamp)
UI
- Holdings‑tabel med sticky header, zebra, badges for status (gain/loss)
- Summary‑cards (Total value, Day P/L, Unrealized P/L, Positions)
- Sidepanel/modal til Add/Edit position; simpel validering
Tech/UX
- Lokal storage bevares; struktur dokumenteres
- URL‑synk for filtrering/sort
Acceptance criteria
- [ ] CRUD uden console errors; tom‑tilstand klar

### Fase 5 – Alerts (v1 shell)
UI
- Cards for “Create rule” (RSI, buy‑zone touch, price cross)
- Liste over regler; status‑badge; en simpel “dry‑run last N days”
Tech/UX
- Kun UI (ingen backend push) – gemmes lokalt
Acceptance criteria
- [ ] Oprettelse/sletning/enable‑toggle virker og persisterer lokalt

### Fase 6 – Settings
- Tema (System/Light/Dark), Accent style (violet/turkis/begge), Datakilde‑badge toggle, Cache‑clear, Endpoint visning
- Eksperiment: kompakt/komfort spacing toggle
Acceptance criteria
- [ ] Indstillinger persisteres lokalt og reflekteres i UI

### Fase 7 – Global polish
- Command Palette (Ctrl/Cmd+K) til jump til tickers/sektioner
- Toasts for handlinger (portfolio add, refresh ok)
- Bedre tom‑/error‑komponenter (ens across app)
- Ikoner (Lucide) harmoniseret
Acceptance criteria
- [ ] 10 vigtigste flows har non‑blocking feedback (toast)

### Fase 8 – A11y & i18n readiness
- Fokus‑ringe, ARIA labels på interaktive kontroller
- Kontrast ≥ 4.5:1 hvor relevant
- Tekst i komponenter samles så i18n kan tilføjes senere
Acceptance criteria
- [ ] Keyboard only navigation gennemføres for Top 3 flows

### Fase 9 – Performance
- Bundle analyse; dead code trim; memoization for tunge tabeller
- Netværk: cache‑buster kun ved refresh; ellers etag/if‑none‑match (senere)
- Lighthouse mål: Perf ≥ 90, Acc ≥ 95, Best‑practices ≥ 95
Acceptance criteria
- [ ] Web Vitals i dev: LCP < 2.5s, CLS ~0, INP < 200ms (lokal test)

### Fase 10 – QA & Release
- Visuelle snapshots (Playwright/Storybook – kræver accept af nye deps)
- E2E røgsignal for primære flows (load board, öppn detail, add portfolio)
- Release notes + toggled rollout hvis nødvendigt
Acceptance criteria
- [ ] PR‑checkliste opfyldt og demo URL postet i PR

---

## Arbejdsgang og branch‑strategi
- Feature branches per fase/underopgave: `feat/ui-<fase>-<kort-navn>`
- Små PRs (≤400 LOC diff). Screenshots/gifs i PR‑beskrivelsen
- Merge til `main` efter visuel review og hurtig røgsignal
- Pages deploy efter merge (automatisk som i dag)

## Definition of Done (per opgave)
- [ ] Acceptance criteria opfyldt
- [ ] Ingen konsol‑fejl/advarsler i devtools
- [ ] Dark mode ser korrekt ud (kontrast + hover/focus‑states)
- [ ] Responsive tjek (≥1440, 1280, 1024, 768, 375)
- [ ] Loading/Empty/Error‑tilstande findes
- [ ] Screenshots lagt i PR

## Målbare succeskriterier
- Lighthouse: Perf ≥ 90, Acc ≥ 95, BP ≥ 95
- Brugeroplevelse: Færre klik for de vigtigste flows, tydeligere hierarki
- Diagnose: Endpoint/Provider badges synlige; Refresh viser nye tickers straks

## Åbne beslutninger
- Command Palette og visuelle regressionstests kræver nye dependencies – afklar før implementering
- Om vi vil have “glass” på alle cards eller kun topbar + udvalgte cards

## Bilag – UI byggesten (uden nye deps)
- Layout: DashboardShell (sidebar + topbar) – DONE
- Cards: border, bg-card/60, shadow-sm
- StatCard: label, værdi, delta‑badge
- Pills/Chips: presets, flags, provider/endpoint badges
- Tabel: sticky header, zebra, hover, compact i mobil med hor. scroll

## Næste sprint (foreslået)
- Fase 2 (Leaderboard)
  - [x] StatCards: Tickers/Avg Score/Updated (Provider mix pending)
  - [ ] Chip‑presets + keyboard shortcuts
  - [ ] Tabel‑polish + tom‑tilstand  •  [x] Skeleton rows (loading)
  - [ ] URL‑synk for søgning/sort/preset

---

Feedback: Ret denne roadmap via PR til `docs/UI_ROADMAP.md`.

