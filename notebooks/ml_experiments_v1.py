# %% [markdown]
# ML Experiments v1 (Fase 9)
# 
# Mål: Baseline-modeller på egne data fra Storage (CSV/Parquet) via web-API.
# - Klassifikation: "købszone → efterfølgende +X% inden N dage" (approksimeret som future_max_ret >= X)
# - Eksporterede features (Parquet/CSV) + baseline-metrikker
# - Reproducérbar: kan køres lokalt mod kørende Next.js API (med SUPABASE envs)
# 
# Forudsætninger:
# - Kør web-appen: (cd web && npm run dev)
# - Sæt SUPABASE_URL, SUPABASE_KEY (+ BUCKET) i .env (server)
# - Python deps: pandas, numpy, requests, scikit-learn (valgfri), duckdb (valgfri)

# %%
import os
from pathlib import Path
import json

# Project roots
ROOT = Path(__file__).resolve().parents[1]
API_BASE = os.getenv('API_BASE', 'http://localhost:3000')

# Params
HORIZON = int(os.getenv('ML_HORIZON', '10'))  # N dage
UP = float(os.getenv('ML_UP', '0.05'))        # +X%
DOWN = float(os.getenv('ML_DOWN', '0.05'))    # -Y% (ikke brugt i v1)

# %%
# Indlæs og byg datasæt
from scripts.ml.build_features import build_dataset, export_features, compute_baselines

# Universe fra scripts/tickers.txt
TICKERS_FILE = ROOT / 'scripts' / 'tickers.txt'
TICKERS = [l.strip() for l in TICKERS_FILE.read_text(encoding='utf-8').splitlines() if l.strip() and not l.strip().startswith('#')]

df = build_dataset(API_BASE, TICKERS, horizon=HORIZON, up=UP, down=DOWN)

# %%
# Eksportér features og gem eksempeludsnit
pq, csv = export_features(df)
print('Features skrevet til:', pq, 'og', csv)
df.head(3)

# %%
# Baseline modeller (hvis scikit-learn er installeret)
metrics = compute_baselines(df)
print(json.dumps(metrics, indent=2))
(Path(ROOT / 'data' / 'ml' / 'metrics.json')).write_text(json.dumps(metrics, indent=2), encoding='utf-8')
metrics

# %% [markdown]
# Næste skridt
# - Justér features (tekniske/fundamentale/sentiment), test flere horisonter (N) og tærskler (X/Y)
# - Evaluér alternative modeller (XGBoost/LightGBM) og tidsopdelte/rolling-valideringer
# - Integrér DuckDB til hurtig feature-udtræk fra Parquet direkte (valgfrit)

