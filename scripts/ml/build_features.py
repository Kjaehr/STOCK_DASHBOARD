#!/usr/bin/env python3
"""
Build ML features and labels from the data lake (CSV/Parquet in Supabase Storage) or fallback to live yfinance.
- Reads tickers from scripts/tickers.txt by default
- For each ticker, downloads CSV via the Next.js API /api/download (which returns a signed URL)
- Produces a single features dataset with engineered features and labels for classification/regression
- Exports to data/ml/features.parquet and data/ml/features.csv

Usage examples:
  python scripts/ml/build_features.py \
    --api http://localhost:3000 \
    --tickers-file scripts/tickers.txt \
    --horizon 10 --up 0.05 --down 0.05

Notes:
- This script is safe to import in notebooks. If scikit-learn is available, it can also compute baseline metrics.
- No secrets are embedded. The /api/download route handles signing server-side.
"""
from __future__ import annotations
import argparse
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Optional heavy deps – keep import guarded for environments without ML stack
try:
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    np = None  # type: ignore

# Optional ML baselines
try:
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
except Exception:  # pragma: no cover
    train_test_split = None  # type: ignore
    roc_auc_score = f1_score = average_precision_score = None  # type: ignore
    LogisticRegression = RandomForestClassifier = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ML_DIR = ROOT / "data" / "ml"


def read_tickers(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _download_csv_for_ticker(api_base: str, ticker: str) -> Optional[str]:
    if not requests:
        return None
    try:
        u = f"{api_base.rstrip('/')}/api/download?ticker={ticker}&fmt=csv"
        r = requests.get(u, timeout=30)
        r.raise_for_status()
        signed = r.json().get("url")
        if not signed:
            return None
        csv = requests.get(signed, timeout=60)
        if not csv.ok:
            return None
        return csv.text
    except Exception:
        return None


def _load_csv_to_df(csv_text: str) -> 'pd.DataFrame':
    assert pd is not None
    df = pd.read_csv(io.StringIO(csv_text))
    # expected columns: timestamp_iso, ticker, price, score, rsi, atr_pct, sma20, sma50, sma200, sector, industry, sent_mean7, sent_count7, flags, in_buy_zone, dist_to_stop_pct, dist_to_t1_pct
    # normalize types
    if 'timestamp_iso' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_iso'], utc=True)
        df = df.drop(columns=['timestamp_iso'])
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.upper()
    num_cols = [c for c in df.columns if c not in ('ticker','flags','sector','industry') and c != 'timestamp']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _fallback_build_from_yfinance(ticker: str, days: int = 365) -> Optional['pd.DataFrame']:
    """Minimal fallback builder using yfinance if Storage/API is unavailable."""
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    assert pd is not None, "pandas required"
    try:
        t = yf.Ticker(ticker)
        h = t.history(period='2y', interval='1d', auto_adjust=False)
        if h is None or len(h) == 0:
            return None
        close = h['Close'].astype(float)
        high = h.get('High', close)
        low = h.get('Low', close)
        sma20 = close.rolling(20, min_periods=1).mean()
        sma50 = close.rolling(50, min_periods=1).mean()
        sma200 = close.rolling(200, min_periods=1).mean()
        prev_close = close.shift(1)
        tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        atr_pct = (atr / close) * 100.0
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(close.index, utc=True),
            'ticker': ticker.upper(),
            'price': close.values,
            'sma20': sma20.values,
            'sma50': sma50.values,
            'sma200': sma200.values,
            'atr_pct': atr_pct.values,
        })
        return df
    except Exception:
        return None


def _features_and_labels(df: 'pd.DataFrame', horizon: int, up: float, down: float) -> 'pd.DataFrame':
    assert pd is not None and np is not None
    df = df.sort_values(['ticker','timestamp']).reset_index(drop=True)
    # Basic derived features
    for ma in ('sma20','sma50','sma200'):
        if ma in df.columns:
            df[f'price_over_{ma}'] = df['price'] / df[ma]
    if 'rsi' in df.columns:
        df['rsi_norm'] = df['rsi'] / 100.0
    if 'atr_pct' in df.columns:
        df['atr_bucket'] = pd.cut(df['atr_pct'], bins=[-1,1,2,4,8,1000], labels=[0,1,2,3,4]).astype('Int64')
    # Per-ticker future stats
    def compute_future(group: 'pd.DataFrame') -> 'pd.DataFrame':
        p = group['price']
        # future max/min over next N closes
        fmax = p.iloc[::-1].rolling(window=horizon, min_periods=1).max().iloc[::-1]
        fmin = p.iloc[::-1].rolling(window=horizon, min_periods=1).min().iloc[::-1]
        group['future_max_ret'] = (fmax / p) - 1.0
        group['future_min_ret'] = (fmin / p) - 1.0
        return group
    df = df.groupby('ticker', group_keys=False).apply(compute_future)
    # Labels (approximation): did we reach +X% within N days?
    df['label_hit_up'] = (df['future_max_ret'] >= float(up)).astype('int8')
    # Optional regression target
    df['target_ret_N'] = df['future_max_ret'].astype(float)
    # Drop rows near end with insufficient lookahead (keep but mark if wanted)
    df['valid_horizon'] = (~df['future_max_ret'].isna()).astype('int8')
    return df


def build_dataset(api_base: Optional[str], tickers: List[str], horizon: int, up: float, down: float) -> 'pd.DataFrame':
    assert pd is not None, "pandas required. Install: pip install pandas numpy requests scikit-learn duckdb"
    frames: List['pd.DataFrame'] = []
    for t in tickers:
        csv_txt = None
        if api_base:
            csv_txt = _download_csv_for_ticker(api_base, t)
        if csv_txt:
            df = _load_csv_to_df(csv_txt)
        else:
            df = _fallback_build_from_yfinance(t)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError("No data loaded. Ensure the web app is running with SUPABASE envs, or install yfinance.")
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=['price'])
    out = _features_and_labels(full, horizon=horizon, up=up, down=down)
    return out


def export_features(df: 'pd.DataFrame') -> Tuple[Path, Path]:
    ML_DIR.mkdir(parents=True, exist_ok=True)
    pq = ML_DIR / 'features.parquet'
    csv = ML_DIR / 'features.csv'
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        pass
    df.to_csv(csv, index=False)
    return pq, csv


def compute_baselines(df: 'pd.DataFrame') -> dict:
    if LogisticRegression is None or train_test_split is None:
        return {"note": "sklearn not installed; skipping baselines"}
    # Minimal feature set (drop leakage/targets)
    cols_drop = {'timestamp','ticker','future_max_ret','future_min_ret','valid_horizon','label_hit_up','target_ret_N','flags','sector','industry'}
    X = df.drop(columns=[c for c in cols_drop if c in df.columns]).select_dtypes(include=['number']).fillna(0.0)
    y = df['label_hit_up'].astype(int)
    # Time-aware split: last 20% as test
    df_sorted = df.sort_values('timestamp')
    n = len(df_sorted)
    split_idx = int(n * 0.8)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    res = {}
    # Logistic Regression
    try:
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, y_train)
        ps = lr.predict_proba(X_test)[:,1]
        res['logreg_auc'] = float(roc_auc_score(y_test, ps)) if roc_auc_score else None
        res['logreg_ap'] = float(average_precision_score(y_test, ps)) if average_precision_score else None
    except Exception as e:
        res['logreg_error'] = str(e)
    # Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        rf.fit(X_train, y_train)
        ps = rf.predict_proba(X_test)[:,1]
        res['rf_auc'] = float(roc_auc_score(y_test, ps)) if roc_auc_score else None
        res['rf_ap'] = float(average_precision_score(y_test, ps)) if average_precision_score else None
    except Exception as e:
        res['rf_error'] = str(e)
    res['n_train'] = int(len(train_idx))
    res['n_test'] = int(len(test_idx))
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--api', default=os.getenv('API_BASE', 'http://localhost:3000'), help='Base URL for web app (for /api/download)')
    p.add_argument('--tickers-file', default=str(ROOT / 'scripts' / 'tickers.txt'))
    p.add_argument('--horizon', type=int, default=10, help='Label horizon in trading days')
    p.add_argument('--up', type=float, default=0.05, help='+X% threshold for positive label')
    p.add_argument('--down', type=float, default=0.05, help='-Y% threshold (reserved for future use)')
    p.add_argument('--no-baseline', action='store_true', help='Skip baseline training metrics')
    args = p.parse_args()

    tickers = read_tickers(Path(args.tickers_file))
    if not tickers:
        print('No tickers found. Create scripts/tickers.txt')
        return 1

    api_base = args.api or None
    try:
        df = build_dataset(api_base, tickers, horizon=args.horizon, up=args.up, down=args.down)
    except Exception as e:
        print(f"ERROR: data load failed: {e}")
        return 2

    pq, csv = export_features(df)
    print(f"Wrote features: {pq if pq.exists() else '(parquet skipped)'} and {csv}")

    if not args.no_baseline:
        metrics = compute_baselines(df)
        (ML_DIR / 'metrics.json').write_text(__import__('json').dumps(metrics, indent=2), encoding='utf-8')
        print('Baseline metrics saved to data/ml/metrics.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

