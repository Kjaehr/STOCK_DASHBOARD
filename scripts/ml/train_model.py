#!/usr/bin/env python3
"""
Train a simple logistic regression using existing screener features and export a lightweight
model artifact compatible with web/src/lib/ml/model.ts and features.ts.

Data source:
- Reads ./data/meta.json (tickers list) and ./data/<TICKER>.json for features that mirror
  toBaseFeatures() in web/src/lib/ml/features.ts

Label (temporary baseline until we have proper future-return labels):
- y = 1 if (score >= 60) else 0

Artifact schema (JSON):
{
  "version": "v1_YYYYMMDDHHMM",
  "features": ["price_over_sma20", "price_over_sma50", ...],
  "intercept": -0.123,
  "coef": [c1, c2, ...],
  "norm": {
    "mean": { "price_over_sma20": 1.02, ... },
    "std":  { "price_over_sma20": 0.15, ... }
  }
}

Usage:
  python scripts/ml/train_model.py --out-dir ml_out --version v1
This writes:
  - ml_out/model_v1_YYYYMMDDHHMM.json
  - ml_out/latest.json (pointer file: {"path": "ml/models/model_v1_YYYYMMDDHHMM.json", "version": "..."})
And prints the artifact filenames so CI can upload them to Supabase Storage.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'

FEATURE_ORDER = [
    'price_over_sma20',
    'price_over_sma50',
    'price_over_sma200',
    'rsi_norm',
    'atr_pct',
    'atr_bucket_0',
    'atr_bucket_1',
    'atr_bucket_2',
    'atr_bucket_3',
    'atr_bucket_4',
    'fcf_yield',
    'nd_to_ebitda',
    'revenue_growth',
    'gross_margin',
    'sent_mean7',
]


def num(x: Any) -> float | None:
    try:
        n = float(x)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def bucket(v: float | None, edges: List[float]) -> int | None:
    if v is None or not math.isfinite(v):
        return None
    for i in range(len(edges) - 1):
        if v >= edges[i] and v < edges[i + 1]:
            return i
    return None


def to_base_features(x: Dict[str, Any]) -> Dict[str, float]:
    """Mirror of web/src/lib/ml/features.ts::toBaseFeatures"""
    f: Dict[str, float] = {}
    price = num(x.get('price'))
    t = x.get('technicals') or {}
    fundamentals = x.get('fundamentals') or {}
    sentiment = x.get('sentiment') or {}

    sma20 = num(t.get('sma20'))
    sma50 = num(t.get('sma50'))
    sma200 = num(t.get('sma200'))
    rsi = num(t.get('rsi'))
    atr_pct = num(t.get('atr_pct'))

    f['price_over_sma20'] = (price / sma20) if (price is not None and sma20 not in (None, 0)) else 1.0
    f['price_over_sma50'] = (price / sma50) if (price is not None and sma50 not in (None, 0)) else 1.0
    f['price_over_sma200'] = (price / sma200) if (price is not None and sma200 not in (None, 0)) else 1.0
    f['rsi_norm'] = (rsi if rsi is not None else 50.0) / 100.0
    f['atr_pct'] = atr_pct if atr_pct is not None else 0.0

    b = bucket(atr_pct, [-math.inf, 1, 2, 4, 8, math.inf])
    for i in range(5):
        f[f'atr_bucket_{i}'] = 1.0 if b == i else 0.0

    fcf_yield = num(fundamentals.get('fcf_yield'))
    nd_to_ebitda = num(fundamentals.get('nd_to_ebitda'))
    revenue_growth = num(fundamentals.get('revenue_growth'))
    gross_margin = num(fundamentals.get('gross_margin'))
    sent7 = num(sentiment.get('mean7'))

    f['fcf_yield'] = fcf_yield if fcf_yield is not None else 0.0
    f['nd_to_ebitda'] = nd_to_ebitda if nd_to_ebitda is not None else 0.0
    f['revenue_growth'] = revenue_growth if revenue_growth is not None else 0.0
    f['gross_margin'] = gross_margin if gross_margin is not None else 0.0
    f['sent_mean7'] = sent7 if sent7 is not None else 0.0

    return f


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    meta_path = DATA_DIR / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Ensure 'data/' is present in repo.")
    meta = json.loads(meta_path.read_text())
    tickers = meta.get('tickers') or []
    X: List[List[float]] = []
    y: List[int] = []
    used: List[str] = []
    for t in tickers:
        p = DATA_DIR / f"{str(t).replace(' ', '_')}.json"
        if not p.exists():
            continue
        row = json.loads(p.read_text())
        # label: 1 if score >= 60 else 0 (baseline)
        score = row.get('score')
        if score is None:
            continue
        label = 1 if float(score) >= 60 else 0
        f = to_base_features(row)
        X.append([float(f.get(k, 0.0)) for k in FEATURE_ORDER])
        y.append(label)
        used.append(str(row.get('ticker') or t))

    if not X:
        raise RuntimeError('No training samples found')
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int32), used


def standardize(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std_safe = np.where(std == 0, 1.0, std)
    Xs = (X - mean) / std_safe
    mean_map = {FEATURE_ORDER[i]: float(mean[i]) for i in range(len(FEATURE_ORDER))}
    std_map = {FEATURE_ORDER[i]: float(std_safe[i]) for i in range(len(FEATURE_ORDER))}
    return Xs, mean_map, std_map


def train_and_dump(out_dir: Path, version_prefix: str = 'v1') -> Tuple[Path, Path]:
    X, y, used = load_dataset()
    Xs, mean_map, std_map = standardize(X)

    clf = LogisticRegression(max_iter=200, solver='liblinear')
    clf.fit(Xs, y)

    coef = clf.coef_.reshape(-1).tolist()
    intercept = float(clf.intercept_.reshape(-1)[0])

    ts = datetime.utcnow().strftime('%Y%m%d%H%M')
    version = f"{version_prefix}_{ts}"

    artifact = {
        'version': version,
        'features': FEATURE_ORDER,
        'intercept': intercept,
        'coef': [float(c) for c in coef],
        'norm': {
            'mean': mean_map,
            'std': std_map,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"model_{version}.json"
    model_path = out_dir / model_filename
    model_path.write_text(json.dumps(artifact, indent=2))

    # pointer file for latest.json in Storage
    latest = {
        'path': f"ml/models/{model_filename}",
        'version': version,
    }
    latest_path = out_dir / 'latest.json'
    latest_path.write_text(json.dumps(latest, indent=2))

    print(str(model_path))
    print(str(latest_path))
    return model_path, latest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'ml_out'))
    ap.add_argument('--version', default='v1')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    train_and_dump(out_dir, args.version)


if __name__ == '__main__':
    main()

