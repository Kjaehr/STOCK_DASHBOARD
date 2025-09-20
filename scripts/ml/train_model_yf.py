#!/usr/bin/env python3
"""
Train logistic regression on yfinance time series with event-style labels:
Label = 1 if price hits +UP% before -DOWN% within HORIZON days after entry day, else 0.

Outputs a lightweight artifact JSON compatible with web/src/lib/ml/{model,features}.ts

Defaults:
- years = 3 (lookback period)
- horizon = 20 trading days
- up = 0.05 (5%)
- down = 0.05 (5%)
- max_tickers = 30 (to keep CI runtime reasonable)

Example:
  python scripts/ml/train_model_yf.py --out-dir ml_out --years 3 --horizon 20 --up 0.05 --down 0.05
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any
import math
import json
import random
import time
from io import StringIO
import re
import logging
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
import requests

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
    # Enhanced technical features
    'vol20_rising',
    'price_gt_ma20',
    'rsi_oversold',
    'rsi_overbought',
    'sma_alignment',
    'above_all_smas',
]


# Shared requests session with a browser-like User-Agent for Yahoo endpoints
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
})
# Reduce yfinance log noise in CI
logging.getLogger('yfinance').setLevel(logging.ERROR)



# --- Yahoo CSV (crumb) helpers ---

def _yahoo_get_crumb_and_cookies(symbol: str) -> tuple[str | None, dict]:
    url = f"https://finance.yahoo.com/quote/{symbol}/history?p={symbol}"
    r = SESSION.get(url, timeout=15)
    if r.status_code != 200 or not r.text:
        return None, {}
    m = re.search(r'CrumbStore\":\{\"crumb\":\"(.*?)\"\}', r.text)
    if not m:
        # Sometimes present without extra escapes
        m = re.search(r'"crumb":"(.*?)"', r.text)
    if not m:
        return None, dict(r.cookies)
    crumb = m.group(1)
    # Unescape common sequences
    crumb = crumb.replace('\\u002F', '/').replace('\\u003D', '=')
    return crumb, dict(r.cookies)


def fetch_history_yahoo_csv(ticker: str, years: int, attempts: int = 2) -> pd.DataFrame | None:
    """Fetch daily OHLC from Yahoo CSV download with crumb+cookies. Auto-adjust OHLC via Adj Close factor."""
    # Compute periods
    end = int(pd.Timestamp.utcnow().timestamp())
    start_ts = pd.Timestamp.utcnow() - pd.DateOffset(years=years)
    start = int(start_ts.timestamp())
    last_err = None
    for a in range(attempts):
        try:
            crumb, cookies = _yahoo_get_crumb_and_cookies(ticker)
            params = {
                'period1': str(start),
                'period2': str(end),
                'interval': '1d',
                'events': 'history',
                'includeAdjustedClose': 'true',
            }
            if crumb:
                params['crumb'] = crumb
            q = '&'.join(f"{k}={quote_plus(v)}" for k, v in params.items())
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?{q}"
            r = SESSION.get(url, cookies=cookies or None, timeout=20, allow_redirects=True)
            if r.status_code != 200 or not r.text or r.text.lstrip().startswith('<'):
                last_err = f"status={r.status_code}"
                time.sleep(1.0 + 0.5 * a)
                continue
            df = pd.read_csv(StringIO(r.text))
            if df is None or df.empty or 'Close' not in df.columns:
                last_err = 'empty/invalid CSV'
                time.sleep(0.5)
                continue
            # Parse and index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']).set_index('Date').sort_index()
            # Auto-adjust OHLC using Adj Close factor if present
            if 'Adj Close' in df.columns and 'Close' in df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    factor = (df['Adj Close'] / df['Close']).replace([np.inf, -np.inf], np.nan)
                for c in ['Open', 'High', 'Low', 'Close']:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce') * factor
                # Prefer adjusted close for Close
                df['Close'] = df['Adj Close']
            # Keep required columns
            keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[keep].dropna(subset=['Open', 'High', 'Low', 'Close'])
            if df.empty:
                last_err = 'dropna empty'
                continue
            return df
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0 + 0.5 * a)
            continue
    if last_err:
        print(f"[yahoo_csv] failed for {ticker}: {last_err}")
    return None

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.bfill()


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # True range
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    pct = (atr / df['Close']) * 100.0
    return pct


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Basic technical features
    sma20 = df['Close'].rolling(20, min_periods=20).mean()
    sma50 = df['Close'].rolling(50, min_periods=50).mean()
    sma200 = df['Close'].rolling(200, min_periods=200).mean()

    out['price_over_sma20'] = df['Close'] / sma20
    out['price_over_sma50'] = df['Close'] / sma50
    out['price_over_sma200'] = df['Close'] / sma200

    rsi_values = rsi(df['Close']).fillna(50.0)
    out['rsi_norm'] = rsi_values / 100.0
    out['atr_pct'] = atr_pct(df).fillna(0.0)

    # ATR buckets: [-inf,1),[1,2),[2,4),[4,8),[8,inf)
    edges = [-math.inf, 1, 2, 4, 8, math.inf]
    def bucketize(x: float) -> int:
        for i in range(5):
            if x >= edges[i] and x < edges[i+1]:
                return i
        return 0
    b = out['atr_pct'].apply(bucketize)
    for i in range(5):
        out[f'atr_bucket_{i}'] = (b == i).astype(float)

    # Enhanced technical features
    vol20 = df['Volume'].rolling(20, min_periods=20).mean()
    vol20_prev = vol20.shift(1)
    out['vol20_rising'] = (vol20 > vol20_prev).astype(float)
    out['price_gt_ma20'] = (df['Close'] > sma20).astype(float)

    # Momentum features
    out['rsi_oversold'] = (rsi_values < 30).astype(float)
    out['rsi_overbought'] = (rsi_values > 70).astype(float)

    # Price position features
    out['sma_alignment'] = ((sma20 > sma50) & (sma50 > sma200)).astype(float)
    out['above_all_smas'] = ((df['Close'] > sma20) & (df['Close'] > sma50) & (df['Close'] > sma200)).astype(float)

    return out


def label_future_path(df: pd.DataFrame, horizon: int, up: float, down: float) -> pd.Series:
    # For each day, look ahead up to horizon days: if High reaches +up% before Low hits -down%, label=1 else 0
    highs = df['High'].values
    lows = df['Low'].values
    close = df['Close'].values
    n = len(df)
    y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        entry = close[i]
        if not math.isfinite(entry) or entry <= 0:
            y[i] = 0
            continue
        up_level = entry * (1.0 + up)
        down_level = entry * (1.0 - down)
        hit = 0
        for j in range(1, horizon + 1):
            k = i + j
            if k >= n:
                break
            if highs[k] >= up_level:
                hit = 1
                break
            if lows[k] <= down_level:
                hit = 0
                break
        y[i] = hit
    return pd.Series(y, index=df.index)


def fetch_history(ticker: str, years: int, attempts: int = 3, pause: float = 1.5) -> pd.DataFrame | None:
    """Robust fetch using Yahoo first (CSV + crumb), then yfinance, then Stooq as last resort."""
    # Small delay to be respectful to Yahoo Finance servers
    time.sleep(0.1)

    # 1) Try Yahoo CSV download (often more reliable in CI if crumb+cookie is set)
    df_csv = fetch_history_yahoo_csv(ticker, years, attempts=2)
    if df_csv is not None and not df_csv.empty:
        print(f"[fetch_history] Using Yahoo CSV for {ticker}: {len(df_csv)} rows")
        return df_csv

    # 2) Fallback to yfinance download with retries
    for a in range(attempts):
        try:
            df = yf.download(
                ticker,
                period=f"{years}y",
                interval='1d',
                auto_adjust=True,
                progress=False,
                session=SESSION,
                group_by='column',
                threads=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                cols = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
                if len(cols) >= 4:
                    out = df[cols].dropna()
                    if not out.empty:
                        print(f"[fetch_history] Using yfinance for {ticker}: {len(out)} rows")
                        return out
        except Exception as e:
            # yfinance sometimes returns HTML/JSONDecodeError; retry with backoff
            if "JSONDecodeError" in str(type(e)) or "Expecting value" in str(e):
                print(f"Failed to get ticker '{ticker}' reason: {e}")
            # Suppress other common yfinance errors to reduce noise
        time.sleep(pause * (a + 1))

    # 3) Last resort: Stooq daily CSV
    try:
        print(f"[fetch_history] Yahoo failed for {ticker}; trying Stooq")
        df_s = fetch_history_stooq(ticker, years)
        if df_s is not None and not df_s.empty:
            print(f"[fetch_history] Using Stooq for {ticker}: {len(df_s)} rows")
            return df_s
    except Exception as e:
        print(f"[fetch_history] Stooq fallback error for {ticker}: {e}")
    print(f"[fetch_history] No data for {ticker} from Yahoo or Stooq")
    return None



def fetch_history_stooq(ticker: str, years: int) -> pd.DataFrame | None:
    """Fetch OHLC from Stooq CSV as a lightweight fallback (no extra deps).
    Stooq US symbols are typically ticker.us (lowercase). Returns last N years.
    """
    s = ticker.lower()
    candidates = []
    if not s.endswith('.us'):
        candidates.append(f"{s}.us")
    candidates.append(s)

    for sym in candidates:
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        try:
            r = SESSION.get(url, timeout=10)
            txt = r.text if r and r.status_code == 200 else ''
            if not txt or txt.lstrip().startswith('<'):  # HTML error page
                continue
            df = pd.read_csv(StringIO(txt))
            if df is None or df.empty:
                continue
            if 'Date' not in df.columns or 'Close' not in df.columns:
                continue
            keep_cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[keep_cols].dropna()
            if df.empty:
                continue
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date').sort_index()
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(years=years)
            df = df[df.index >= cutoff]
            # ensure numeric types
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            if not df.empty:
                return df
        except Exception:
            continue
    return None


def build_dataset(tickers: List[str], years: int, horizon: int, up: float, down: float, max_tickers: int) -> Tuple[np.ndarray, np.ndarray]:
    X_rows: List[List[float]] = []
    Y_rows: List[int] = []
    used = 0
    random.seed(42)
    tickers_shuffled = tickers[:]
    random.shuffle(tickers_shuffled)
    for t in tickers_shuffled:
        if used >= max_tickers:
            break
        df = fetch_history(t, years)
        if df is None or df.empty:
            print(f"[dataset] no data for {t}; skipping")
            continue
        feats = build_features(df)
        lbl = label_future_path(df, horizon=horizon, up=up, down=down)
        # Align: drop rows with NaNs in features (due to warmup)
        Z = feats.join(lbl.rename('y')).dropna()
        if Z.empty:
            print(f"[dataset] features empty after warmup for {t}; skipping")
            continue
        # Use data that still allows a full horizon window
        # Drop the last horizon rows where label would be truncated by series end
        if len(Z) > horizon:
            Z = Z.iloc[:-horizon]
        else:
            print(f"[dataset] not enough rows after horizon trim for {t}; skipping")
            continue
        X_rows.extend(Z[FEATURE_ORDER].astype(float).values.tolist())
        Y_rows.extend(Z['y'].astype(int).values.tolist())
        used += 1
        print(f"[dataset] used {t} samples={len(Z)} (tickers used={used}/{max_tickers})")
    if not X_rows:
        raise RuntimeError('No samples built from available data sources (Yahoo/Stooq)')
    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(Y_rows, dtype=np.int32)
    return X, y


def standardize(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std_safe = np.where(std == 0, 1.0, std)
    Xs = (X - mean) / std_safe
    mean_map = {FEATURE_ORDER[i]: float(mean[i]) for i in range(len(FEATURE_ORDER))}
    std_map = {FEATURE_ORDER[i]: float(std_safe[i]) for i in range(len(FEATURE_ORDER))}
    return Xs, mean_map, std_map


def train_and_dump(out_dir: Path, years: int, horizon: int, up: float, down: float, max_tickers: int, version_prefix: str = 'v1') -> Tuple[Path, Path]:
    # Tickers from repo data/meta.json to align with app universe
    meta_path = DATA_DIR / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    tickers = [str(t) for t in (meta.get('tickers') or [])]
    if not tickers:
        raise RuntimeError('No tickers in meta.json')

    X, y = build_dataset(tickers, years=years, horizon=horizon, up=up, down=down, max_tickers=max_tickers)
    Xs, mean_map, std_map = standardize(X)

    uniq = np.unique(y)
    if uniq.size < 2:
        eps = 0.01
        p = float(np.clip(y.mean() if y.size > 0 else 0.5, eps, 1 - eps))
        intercept = float(np.log(p / (1 - p)))
        coef = [0.0 for _ in FEATURE_ORDER]
        print(f"[train_model_yf] Single-class dataset detected. Using prior-only model with p={p:.3f}")
    else:
        clf = LogisticRegression(max_iter=250, solver='liblinear', class_weight='balanced')
        clf.fit(Xs, y)
        coef = clf.coef_.reshape(-1).tolist()
        intercept = float(clf.intercept_.reshape(-1)[0])

    ts = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
    version = f"{version_prefix}_{ts}_yf"

    artifact = {
        'version': version,
        'features': FEATURE_ORDER,
        'intercept': float(intercept),
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
    ap.add_argument('--years', type=int, default=3)
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--up', type=float, default=0.05)
    ap.add_argument('--down', type=float, default=0.05)
    ap.add_argument('--max-tickers', type=int, default=30)
    ap.add_argument('--version', default='v1')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    train_and_dump(out_dir, years=args.years, horizon=args.horizon, up=args.up, down=args.down, max_tickers=args.max_tickers, version_prefix=args.version)


if __name__ == '__main__':
    main()

