Klart svar: du kan få mærkbar accuracy-løft med bedre mål + bedre features + bedre træningsregime. Her er en skarp, prioriteret plan med konkrete greb, du kan implementere trin for trin.

1) Mål/labels (størst effekt)

Regime-afhængige labels: Del datasættet i regimer (bull/bear/sideways) ud fra f.eks. 60d trend + volatilitet. Træn én model pr. regime eller giv regimet som feature. Det løfter både kalibrering og accuracy.

Volatilitets-normaliseret horizon: Lad horizon = k * ATR20 i stedet for faste 20 bars. Det reducerer label-støj i stille/volatile perioder.

Asymmetrisk cost / class weights: Brug scale_pos_weight (eller class weights) pr. fold så klasser balanceres — mål “balanced accuracy” og optimer thresholds per klasse i stedet for fixed 0.33/0.5.

Meta-labeling (Lopez de Prado): Behold din primary signalmodel; træn en meta-model (logistic/xgb) til at forudsige om primærsignalets trade bliver profitabelt. Brug meta-proba til at filtrere trades → højere hit-rate.

2) Features (høj ROI, lav kompleksitet)

Trend/momentum

RSI(2), RSI(14), Stoch %K/%D, ROC(5/10/20), MOM(10/20).

Slope-features: hældning af SMA20/50/200 over de seneste 10 bars.

Breakout-tryktest: distance til 20/55-d høj/lav, samt antal forsøg på brud (antal gange prisen har været indenfor 0.5*ATR af niveauet de sidste 10/20 bars).

Volatilitet/risiko

ATR(14), ATR% = ATR/Close, HV(10/20) (hist. volatilitet), True Range til range ratio.

Volatility squeeze: BBWidth(20) og Keltner-kanal bredde; z-score af BBWidth.

Volume/flow

OBV + OBV-slope, ADL (Accum/Dist).

Volume-zscore: (vol - SMA20(vol))/std20(vol).

Up/Down volume spread: diff mellem vol på op-dage vs. ned-dage over 5/10 bars.

Relative/markedsbreddde

Relativ styrke mod benchmark (SPY/QQQ/sector ETF): RS = Close_ticker / Close_benchmark + RS-slope.

Internals proxy (hvis du ikke vil hente nye dataserier): lav et “market risk” feature som 1) benchmark drawdown-zscore, 2) VIX-proxy via SPY intraday range.

Mean-reversion / mikrostruktur

K-eltner/BB pos: position i bånd [0..1].

Wick-features: (High-Close)/(High-Low) og (Close-Low)/(High-Low).

Gap-features: Overnight gap i % og efterfølgende fill-rate (sidste 5 forekomster).

Kvalitet/temporal

Age-features: dage siden 52w high/low; dage siden sidste earnings.

Tickers som kategori: giv ticker-embedding/one-hot (eller sector/industry) så modellen lærer forskelle mellem biotech/tech osv.

3) Event- og kalenderfeatures (middel ROI, stor effekt for accuracy)

Earnings-nærhed: days_to_earnings og is_earnings_window (±3d).

Makro-kalender (CPI/FOMC/ECB/Non-Farm): simple dummies for næste 3 handelsdage.

Short-term news sentiment: rullende 1–3 d sentiment score (selv bare en simpel finBERT på overskrifter kan flytte accuracy).

4) Datakvalitet & lækagekontrol (gratis accuracy)

Stram purged CV: hold embargo ≥ horizon; hvis du allerede bruger 20 bars, test 30–40.

Target alignment audit: verificér at alle features er kun baseret på info ≤ t-1 (ingen same-bar beregning efter close hvis target er intrabar).

Survivorship bias: med microcaps — sikre delisted/IPO-hændelser er håndteret (ellers overoptimisme).

5) Træningsregime

Time-decay vægte: vægt nyere observationer højere (w = exp(-Δt / τ)); sænker koncept drift.

Focal loss (XGBoost custom): hjælper mod “nemme” eksempler og ubalancer → højere accuracy på de svære klasser.

Stacked ensemble: behold RF + XGB + Logistic, men træn en Meta-LR/XGB på [p_rf, p_xgb, p_logit, regime, vol%]. Brug K-fold OOF-proba til at undgå lækage.

Calibrering per regime: Platt/Isotonic pr. regime forbedrer beslutningstærskler.

6) Threshold-optimering (hurtig gevinst)

I stedet for argmax, grid-søg tærskler på validering for at maksimere balanced accuracy eller MCC. Gem thresholds per fold og gennemsnit dem. Det kan alene give +2–5 pct.point.

7) Feature selection uden at kaste guld ud

Ablation-suite: kør automatiske eksperimenter: tilføj én featuregruppe ad gangen (trend → vol → volume → RS → events). Behold kun grupper der giver stabil gevinst ≥ +0.5 pct.point på out-of-fold.

SHAP-screen: brug SHAP til at opdage redundans og non-linear interaktioner, men træf beslutninger via purged OOF-ablation (ikke kun SHAP).

8) Hyperparam-tuning (målrettet)

XGB: søg i max_depth 4–8, min_child_weight 5–20, subsample 0.6–0.95, colsample_bytree 0.5–0.9, lambda 0.5–10, alpha 0–1, n_estimators 300–900. Brug early_stopping_rounds 100 med en val-split indenfor folden.

RF: færre, dybere træer kan overfitte; prøv n_estimators 300–600, max_depth 8–14, min_samples_leaf 20–80; mål MCC/balanced accuracy.

Meta-model: hold den simpel (Logistic/XGB med lav dybde) — du vil have stabilitet, ikke overfit.

9) Stabil inferens

Feature drift monitor: track distributions pr. feature pr. måned; alarm hvis KS-distance > tærskel.

Re-kalibrering: genkalibrér hver 1–3 mdr. på de nyeste 3–6 mdr. uden at retræne hele modellen.

10) Minimal kodeændring (kan laves nu)

Tilføj disse 8 features straks (lav ROI-prøve):

ATR14%, BBWidth20_z, RS_slope_20, OBV_slope_10,

WickTop = (High-Close)/(High-Low+1e-9),

WickBot = (Close-Low)/(High-Low+1e-9),

GapPct = (Open - PrevClose)/PrevClose,

DaysSince52wHigh/Low.

Tilføj regime = sign(SMA50 - SMA200) sammen med vol_regime = quantile(ATR14%).

Indfør threshold-grid per fold og gem bedste thresholds.