# ML Baselines (Fase 9)

Mål: Første, reproducerbare ML‑eksperimenter på egne data via Storage (CSV/Parquet) og en simpel notebook/script‑pipeline.

Indhold:
- notebooks/ml_experiments_v1.py (jupytext‑venlig notebook‑stil)
- scripts/ml/build_features.py (CLI + importérbar utils)
- data/ml/features.parquet og data/ml/features.csv (output)
- data/ml/metrics.json (baseline‑metrikker, hvis sklearn er installeret)

Forudsætninger
- Kør Next.js API lokalt: (cd web && npm install && npm run dev)
- Sæt .env for web med: SUPABASE_URL, SUPABASE_KEY (eller SERVICE_ROLE), SUPABASE_BUCKET
- Python: pip install pandas numpy requests scikit-learn duckdb (duckdb er valgfri i v1)

Kørsel (hurtigstart)
1) Start API: web på http://localhost:3000
2) Bekræft tickers i scripts/tickers.txt
3) Kør feature‑bygning og baselines fra repo‑roden:
   
   python scripts/ml/build_features.py --api http://localhost:3000 --tickers-file scripts/tickers.txt --horizon 10 --up 0.05 --down 0.05

   Output:
   - data/ml/features.parquet (hvis pyarrow/fastparquet er installeret, ellers kun CSV)
   - data/ml/features.csv
   - data/ml/metrics.json (AUC/AP for LogReg/RF; hvis sklearn findes)

Notebook
- Åbn notebooks/ml_experiments_v1.py i VS Code (som notebook) eller kør celler med jupytext/interactive window
- Notebook loader data via scripts/ml/build_features (samme parametre via env: API_BASE, ML_HORIZON, ML_UP, ML_DOWN)

Labels og features (v1)
- Label (klassifikation): hit_up = 1 hvis future_max_ret >= +X% indenfor N dage (approksimeret, kun dagsluk)
- Regression target: target_ret_N = future_max_ret
- Features: pris vs. SMA20/50/200, RSI normaliseret, ATR% bucket, sentiment/fundamental felter hvis tilgængelige
- Bemærk: Ordnet sekvens ("+X% før −Y%") er ikke modelleret i v1; kan tilføjes med intradag eller mere avanceret logik

Videre arbejde
- DuckDB: læs Parquet direkte (stocks-datalake/parquet/...) for hurtigere joins og historik
- Flere modeller: XGBoost/LightGBM, tidsrullet validering, kalibrering
- Feature store: definér versionsmærkning for feature‑sæt og param‑konfiguration
- Mål: dokumentér forbedringer i denne fil pr. iteration

