#!/usr/bin/env python3
"""
Enhanced ML training with XGBoost, Random Forest, and ensemble methods.
Supports multi-class labels, multiple timeframes, and advanced validation.
Uses robust data fetching with Stooq fallback (no yfinance dependency required).

Example:
  python scripts/ml/train_model_ensemble.py --out-dir ml_out --version v3 --model-type ensemble
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, cast, Literal
import json
import numpy as np
import pandas as pd
import requests
import time
import re
from io import StringIO
import os
import numbers

from urllib.parse import quote_plus
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    joblib = None  # type: ignore[assignment]
    HAS_JOBLIB = False

mlflow = None
try:
    import mlflow  # type: ignore
    HAS_MLFLOW = True
except Exception:
    mlflow = None
    HAS_MLFLOW = False

wandb = None
try:
    import wandb  # type: ignore
    HAS_WANDB = True
except Exception:
    wandb = None
    HAS_WANDB = False

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

xgb = None
try:
    import xgboost as xgb  # type: ignore
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

LGBMClassifier = None
try:
    from lightgbm import LGBMClassifier  # type: ignore
    HAS_LIGHTGBM = True
except ImportError:
    LGBMClassifier = None
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")
# Backward-compat alias for requested flag name
HAS_LGBM = HAS_LIGHTGBM

def clean_lgbm_params(p: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LightGBM params to avoid alias warnings and set safe defaults.
    - Remove colsample_bytree/subsample in favor of feature_fraction/bagging_fraction
    - Ensure bagging_freq=1 and reasonable defaults if missing
    """
    p = dict(p)
    # Remove XGBoost-style aliases if present (LGBM warns otherwise)
    p.pop('colsample_bytree', None)
    p.pop('subsample', None)
    # Set sensible defaults if missing
    p.setdefault('feature_fraction', 0.7)
    p.setdefault('bagging_fraction', 0.8)
    p.setdefault('bagging_freq', 1)
    return p

shap = None
try:
    import shap  # type: ignore
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False
    print("Warning: SHAP not installed. Install with: pip install shap")
optuna = None
try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False
    print("Warning: Optuna not installed. Install with: pip install optuna")

from typing import Any as OptunaTrial  # Optuna Trial type alias (avoids hard import for editors)



REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'

# Shared requests session with browser-like User-Agent
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
})

# --- Data Quality & Leakage Control helpers ---

def _check_suspicious_feature_names(feature_names: List[str]) -> None:
    """Strict check for forward-looking feature names.
    Raises ValueError with actionable guidance when suspicious names are found.
    """
    try:
        import re as _re
    except Exception:
        _re = None  # type: ignore
    suspicious: List[str] = []
    # Expanded patterns covering common leakage indicators
    patterns = [
        r"(ret|return).*?(fwd|forward|lead|t\+\d+)",
        r"^(fwd_|future_|next_)",
        r"_(fwd|lead|t\+\d+)$",
        r"(y_next|y\+\d+|target|label)",
        r"future_(max|min|ret)",
        r"(max|min)_future",
    ]
    for n in feature_names:
        s = str(n)
        for p in patterns:
            if (_re.search(p, s, flags=_re.IGNORECASE) if _re else (p.lower() in s.lower())):
                suspicious.append(s)
                break
    if suspicious:
        hint = (
            "Remove forward-looking columns (e.g., future_*, *_fwd, next_*), and ensure features are built "
            "only from information available at or before t-1. Avoid joining label targets back into features."
        )
        raise ValueError(
            f"Temporal leakage detected in feature names: {suspicious[:10]}. {hint}"
        )


def _assert_purged_no_overlap(
    date_ser: "pd.Series",
    tick_ser: Optional["pd.Series"],
    test_mask: "pd.Series",
    train_mask: "pd.Series",
    emb_left: "pd.Timestamp",
    emb_right: "pd.Timestamp",
) -> None:
    """Assert that training rows do not fall inside the embargo window around the test period.
    If tickers are provided, enforce the purge per ticker; otherwise enforce globally.
    """
    import numpy as _np
    import pandas as _pd
    in_window = (date_ser >= emb_left) & (date_ser <= emb_right)
    if tick_ser is not None:
        viol_total = 0
        test_tickers = _pd.Series(tick_ser[test_mask]).unique()
        for t in test_tickers:
            t_mask = (tick_ser == t)
            viol_total += int(_np.sum((_pd.Series(train_mask).values.astype(bool)) & t_mask.values & in_window.values))
        if viol_total > 0:
            raise AssertionError(f"Purged CV violation: {viol_total} training rows within embargo window [{emb_left}, {emb_right}] for test tickers")
    else:
        viol = int(_np.sum((_pd.Series(train_mask).values.astype(bool)) & in_window.values))
        if viol > 0:
            raise AssertionError(f"Purged CV violation: {viol} training rows within embargo window [{emb_left}, {emb_right}]")




def _compute_dynamic_embargo_bars(
    feature_names: Optional[List[str]],
    X: np.ndarray,
    mask_test: np.ndarray,
    horizon: int,
    dynamic_horizon_k: Optional[float]
) -> Optional[int]:
    """Compute a per-fold dynamic embargo (in bars) based on ATR20-normalized horizon.
    Returns the 90th percentile of per-sample expected bars-to-target for the test fold.
    If ATR-like columns are unavailable or k is None, returns None.
    """
    try:
        if dynamic_horizon_k is None or feature_names is None or X is None or mask_test is None:
            return None
        # Prefer absolute ATR if present; otherwise, use atr_pct
        col_idx = None
        use_pct = False
        if 'atr' in feature_names:
            col_idx = feature_names.index('atr')
        elif 'atr_pct' in feature_names:
            col_idx = feature_names.index('atr_pct')
            use_pct = True
        else:
            return None
        vals_all = X[:, col_idx].astype(float)
        vals_test = X[mask_test, col_idx].astype(float)
        # Robust median for normalization
        med = np.nanmedian(vals_all)
        if not np.isfinite(med) or med <= 0:
            # Fallback: use 1.0 to avoid division errors
            med = 1.0
        scale_test = vals_test / med if med != 0 else vals_test
        # Expected bars capped to a sensible range
        bars = np.round(float(horizon) * float(dynamic_horizon_k) * scale_test).astype(int)
        bars = np.clip(bars, 5, 60)
        bars = bars[np.isfinite(bars)]
        if bars.size == 0:
            return None
        p90 = int(np.percentile(bars, 90))
        return max(1, int(p90))
    except Exception:
        return None

def _try_audit_survivorship(date_ser: "pd.Series", tickers: Optional[List[str]]) -> None:
    """Best-effort survivorship bias audit: if a security master exists, verify sampled rows.
    Expects a CSV at DATA_DIR/security_master.csv with columns: ticker,start_date,end_date (optional).
    Only logs warnings; does not raise.
    """
    if tickers is None or len(tickers) != len(date_ser):
        return
    import pandas as _pd
    from pathlib import Path as _Path
    sm_path = DATA_DIR / "security_master.csv"
    if not _Path(sm_path).exists():
        return
    try:
        sm = _pd.read_csv(sm_path)
        if sm.empty or "ticker" not in sm.columns:
            return
        sm = sm.copy()
        if "start_date" in sm.columns:
            sm["start_date"] = _pd.to_datetime(sm["start_date"], errors="coerce")
        if "end_date" in sm.columns:
            sm["end_date"] = _pd.to_datetime(sm["end_date"], errors="coerce")
        df = _pd.DataFrame({"date": _pd.to_datetime(date_ser.values), "ticker": list(tickers)})
        # Sample to keep it light
        df = df.sample(min(len(df), 500), random_state=42)
        merged = df.merge(sm[[c for c in ["ticker", "start_date", "end_date"] if c in sm.columns]], on="ticker", how="left")
        if "start_date" not in merged.columns:
            return
        # Build an end column robustly (if missing, allow through by using max date)
        if "end_date" in merged.columns:
            end_col = merged["end_date"].copy()
            end_col = end_col.fillna(merged["date"].max())
        else:
            end_col = _pd.Series([merged["date"].max()] * len(merged), index=merged.index)
        alive = (merged["date"] >= merged["start_date"]) & (merged["date"] <= end_col)
        violations = int((~alive).sum())
        if violations > 0:
            print(f"WARN: Survivorship audit: {violations} sampled rows fall outside ticker listing window. Check your universe construction.")
    except Exception as e:
        print(f"WARN: Survivorship audit skipped due to error: {e}")

# Enhanced feature set
FEATURE_ORDER = [
    # Technical indicators
    'price_over_sma20', 'price_over_sma50', 'price_over_sma200',
    'rsi_norm', 'atr_pct',
    'atr_bucket_0', 'atr_bucket_1', 'atr_bucket_2', 'atr_bucket_3', 'atr_bucket_4',
    'vol20_rising', 'price_gt_ma20', 'rsi_oversold', 'rsi_overbought',
    'rsi_momentum', 'sma_alignment', 'above_all_smas', 'vol_ratio',
    'price_momentum_5d', 'price_momentum_10d', 'bb_position', 'volume_trend',

    # New advanced features (to be implemented)
    'macd_signal', 'stoch_k', 'stoch_d', 'williams_r',
    'cci', 'adx', 'momentum_10d', 'momentum_20d',
    'price_vs_vwap', 'volume_sma_ratio', 'volatility_rank',

    # Fundamental features (to be added)
    'pe_ratio_norm', 'pb_ratio_norm', 'debt_to_equity_norm',
    'roe_norm', 'revenue_growth_norm', 'earnings_growth_norm',

    # Sentiment features (to be added)
    'news_sentiment_1d', 'news_sentiment_7d', 'news_sentiment_30d',
    'social_sentiment', 'analyst_sentiment',

    # Sector/Industry features (to be added)
    'sector_momentum', 'industry_momentum', 'relative_strength_sector',
]

# --- Data fetching functions (Stooq-first approach) ---

def fetch_history_stooq(ticker: str, years: int) -> pd.DataFrame | None:
    """Fetch OHLC from Stooq CSV as primary data source (no extra deps).
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
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            if not df.empty:
                print(f"[Stooq] Fetched {ticker} as {sym}: {len(df)} rows")
                return df
        except Exception as e:
            print(f"[Stooq] Error fetching {sym}: {e}")
            continue
    return None

def _yahoo_get_crumb_and_cookies(symbol: str) -> tuple[str | None, dict]:
    """Get Yahoo Finance crumb and cookies for CSV download"""
    url = f"https://finance.yahoo.com/quote/{symbol}/history?p={symbol}"
    r = SESSION.get(url, timeout=15)
    if r.status_code != 200 or not r.text:
        return None, {}
    m = re.search(r'CrumbStore\":\{\"crumb\":\"(.*?)\"\}', r.text)
    if not m:
        m = re.search(r'"crumb":"(.*?)"', r.text)
    if not m:
        return None, dict(r.cookies)
    crumb = m.group(1)
    crumb = crumb.replace('\\u002F', '/').replace('\\u003D', '=')
    return crumb, dict(r.cookies)

def fetch_history_yahoo_csv(ticker: str, years: int, attempts: int = 2) -> pd.DataFrame | None:
    """Fetch daily OHLC from Yahoo CSV download with crumb+cookies as fallback"""
    end = int(pd.Timestamp.utcnow().timestamp())
    start_ts = pd.Timestamp.utcnow() - pd.DateOffset(years=years)
    start = int(start_ts.timestamp())

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
                time.sleep(1.0 + 0.5 * a)
                continue
            df = pd.read_csv(StringIO(r.text))
            if df is None or df.empty or 'Close' not in df.columns:
                time.sleep(0.5)
                continue
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']).set_index('Date').sort_index()
            # Auto-adjust OHLC using Adj Close factor if present
            if 'Adj Close' in df.columns:
                adj_factor = df['Adj Close'] / df['Close']
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in df.columns:
                        df[col] = df[col] * adj_factor
                df = df.drop(columns=['Adj Close'])
            df = df.dropna()
            if not df.empty:
                print(f"[Yahoo CSV] Fetched {ticker}: {len(df)} rows")
                return df
        except Exception as e:
            print(f"[Yahoo CSV] Error for {ticker} (attempt {a+1}): {e}")
            time.sleep(1.0 + 0.5 * a)
    return None

def fetch_history(ticker: str, years: int) -> pd.DataFrame | None:
    """Robust fetch using Stooq first, then Yahoo CSV as fallback"""
    time.sleep(0.1)  # Be respectful to servers

    # 1) Try Stooq first (more reliable, no rate limits)
    df_stooq = fetch_history_stooq(ticker, years)
    if df_stooq is not None and not df_stooq.empty:
        return df_stooq

    # 2) Fallback to Yahoo CSV
    print(f"[fetch_history] Stooq failed for {ticker}; trying Yahoo CSV")
    df_yahoo = fetch_history_yahoo_csv(ticker, years)
    if df_yahoo is not None and not df_yahoo.empty:
        return df_yahoo

    print(f"[fetch_history] No data for {ticker} from Stooq or Yahoo")
    return None

class MultiClassLabeler:
    """Create multi-class labels instead of binary"""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            'strong_up': 0.10,    # >10%
            'weak_up': 0.02,      # 2-10%
            'sideways': 0.02,     # -2% to 2%
            'weak_down': 0.10,    # -10% to -2%
            'strong_down': 0.10   # <-10%
        }

    def create_labels(self, returns: pd.Series) -> pd.Series:
        """Convert returns to multi-class labels"""
        labels = pd.Series(index=returns.index, dtype='category')
        returns_num = pd.to_numeric(returns, errors='coerce').astype('float64')

        labels[returns_num >= self.thresholds['strong_up']] = 'STRONG_UP'
        labels[(returns_num >= self.thresholds['weak_up']) &
               (returns_num < self.thresholds['strong_up'])] = 'WEAK_UP'
        labels[(returns_num >= -self.thresholds['sideways']) &
               (returns_num < self.thresholds['weak_up'])] = 'SIDEWAYS'
        labels[(returns_num >= -self.thresholds['weak_down']) &
               (returns_num < -self.thresholds['sideways'])] = 'WEAK_DOWN'
        labels[returns_num < -self.thresholds['weak_down']] = 'STRONG_DOWN'

        return labels

class EnsembleTrainer:
    """Train ensemble of models with advanced validation"""

    def __init__(self, model_type: str = 'ensemble', tuner: str = 'optuna'):
        self.model_type = model_type
        self.tuner = tuner
        self.models = {}
        self.ensemble = None
        # Global preprocessor: impute missing values once; linear models add their own scaler in a Pipeline
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        # Dynamically configurable attributes (set from main via argparse)
        self.optuna_trials: Optional[int] = None
        self.optuna_timeout: Optional[int] = None
        self.stacking_cv_splits: int = 3
        self.meta_C: float = 0.5
        # Threshold optimization metric: 'balanced_accuracy' or 'mcc'
        self.threshold_metric: str = 'balanced_accuracy'


    def create_models(self) -> Dict[str, Any]:
        """Create individual models for ensemble with better regularization"""
        models = {
            'logistic': Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    random_state=42,
                    max_iter=2000,
                    class_weight='balanced',
                    C=1.0,
                    penalty='l2',
                    solver='lbfgs'
                ))
            ]),
            'random_forest': RandomForestClassifier(
                n_estimators=50,  # Reduced to prevent overfitting
                random_state=42,
                class_weight='balanced',
                max_depth=8,  # Limit depth
                min_samples_split=10,  # Require more samples to split
                min_samples_leaf=5,   # Require more samples in leaf
                max_features='sqrt',  # Use subset of features
                n_jobs=-1
            )
        }

        if HAS_XGBOOST:
            assert xgb is not None
            models['xgboost'] = xgb.XGBClassifier(
                random_state=42,
                n_estimators=50,  # Reduced
                learning_rate=0.05,  # Lower learning rate
                max_depth=4,  # Shallower trees
                min_child_weight=3,  # More regularization
                subsample=0.8,  # Use subset of samples
                colsample_bytree=0.8,  # Use subset of features
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                eval_metric='logloss'
            )

        if HAS_LIGHTGBM:
            assert LGBMClassifier is not None
            _p = dict(
                random_state=42,
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                class_weight='balanced',
                n_jobs=-1
            )
            _p = clean_lgbm_params(_p)
            models['lgbm'] = LGBMClassifier(**_p)

        return models

    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray,
                            model_name: str, model: Any, sample_weight: Optional[np.ndarray] = None) -> Any:
        """Tune hyperparameters using Optuna (if enabled) or GridSearchCV as fallback."""
        # Optuna branch
        if getattr(self, 'tuner', 'grid') == 'optuna' and HAS_OPTUNA:
            try:
                return self.optuna_tune(X, y, model_name, model, sample_weight)
            except Exception as e:
                print(f"Optuna tuning failed for {model_name}: {e}. Falling back to grid.")

        # GridSearch fallback
        tscv = TimeSeriesSplit(n_splits=3)
        param_grids = {
            'logistic': {
                'clf__C': [0.1, 1.0, 10.0],
                'clf__penalty': ['l2'],
                'clf__solver': ['lbfgs']
            },
            'random_forest': {
                'n_estimators': [30, 50, 100],
                'max_depth': [6, 8, 10],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [3, 5, 7]
            }
        }
        if HAS_XGBOOST and model_name == 'xgboost':
            param_grids['xgboost'] = {
                'n_estimators': [30, 50, 100],
                'learning_rate': [0.03, 0.05, 0.1],
                'max_depth': [3, 4, 6],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
        if HAS_LIGHTGBM and model_name == 'lgbm':
            param_grids['lgbm'] = {
                'n_estimators': [100, 200, 400],
                'learning_rate': [0.03, 0.05, 0.1],
                'num_leaves': [15, 31, 63],
                'feature_fraction': [0.6, 0.8, 0.9],
                'bagging_fraction': [0.6, 0.8, 0.9],
                'min_child_samples': [10, 20, 40],
                'reg_lambda': [0.1, 1.0, 5.0],
                'reg_alpha': [0.0, 0.1, 1.0]
            }
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=tscv,
                scoring='balanced_accuracy',
                n_jobs=-1,
                verbose=2,
                error_score='raise'
            )
            try:
                if sample_weight is not None:
                    if isinstance(model, Pipeline):
                        grid_search.fit(X, y, **{'clf__sample_weight': sample_weight})
                    else:
                        grid_search.fit(X, y, **{'sample_weight': sample_weight})
                else:
                    grid_search.fit(X, y)
            except Exception as e:
                print(f"GridSearchCV failed for {model_name}: {e}")
                raise
            return grid_search.best_estimator_
        return model

    def optuna_tune(self, X: np.ndarray, y: np.ndarray, model_name: str, base_model: Any,
                     sample_weight: Optional[np.ndarray] = None) -> Any:
        """Optuna Bayesian tuning with TimeSeriesSplit, pruning, and early stopping."""
        tscv = TimeSeriesSplit(n_splits=3)
        n_trials = int(getattr(self, 'optuna_trials', 30))
        timeout = getattr(self, 'optuna_timeout', None)
        assert optuna is not None
        try:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
            print("Optuna: Using MedianPruner (startup_trials=3, warmup_steps=1)")
        except Exception:
            try:
                pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
                print("Optuna: Falling back to SuccessiveHalvingPruner")
            except Exception:
                pruner = None
                print("Optuna: No pruner available; proceeding without pruning")

        from sklearn.metrics import balanced_accuracy_score as _ba
        def auc_score(estimator, X_val, y_val):
            preds = estimator.predict(X_val)
            return float(_ba(y_val, preds))

        def objective(trial: 'OptunaTrial') -> float:
            # Expanded search spaces
            if model_name == 'logistic':
                C = trial.suggest_float('C', 1e-4, 1e3, log=True)
                model = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(
                        random_state=42, max_iter=2000, class_weight='balanced',
                        penalty='l2', solver='lbfgs', C=C
                    ))
                ])
            elif model_name == 'random_forest':
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_depth = trial.suggest_int('max_depth', 3, 20)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
                choice = trial.suggest_categorical('max_features_kind', ['sqrt', 'log2', 'float'])
                max_features: float | Literal['sqrt', 'log2']
                if choice == 'float':
                    max_features = float(trial.suggest_float('max_features_float', 0.5, 1.0))
                else:
                    # Narrow type to accepted sklearn literals
                    choice_lit: Literal['sqrt', 'log2'] = cast(Literal['sqrt', 'log2'], choice)
                    max_features = choice_lit
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    max_features=max_features, class_weight='balanced', random_state=42, n_jobs=-1)
            elif model_name == 'xgboost' and HAS_XGBOOST:
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
                subsample = trial.suggest_float('subsample', 0.5, 1.0)
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
                gamma = trial.suggest_float('gamma', 0.0, 5.0)
                reg_alpha = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True)
                reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 20.0, log=True)
                assert xgb is not None
                model = xgb.XGBClassifier(
                    random_state=42, n_estimators=n_estimators, learning_rate=learning_rate,
                    max_depth=max_depth, min_child_weight=min_child_weight,
                    subsample=subsample, colsample_bytree=colsample_bytree,
                    gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                    eval_metric='logloss')
            elif model_name == 'lgbm' and HAS_LIGHTGBM:
                n_estimators = trial.suggest_int('n_estimators', 100, 400)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
                num_leaves = trial.suggest_int('num_leaves', 15, 63)
                feature_fraction = trial.suggest_float('feature_fraction', 0.6, 0.9)
                bagging_fraction = trial.suggest_float('bagging_fraction', 0.6, 0.9)
                min_child_samples = trial.suggest_int('min_child_samples', 10, 50)
                reg_alpha = trial.suggest_float('reg_alpha', 0.0, 5.0)
                reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)
                assert LGBMClassifier is not None
                _p = dict(
                    random_state=42,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    feature_fraction=feature_fraction,
                    bagging_fraction=bagging_fraction,
                    min_child_samples=min_child_samples,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    class_weight='balanced',
                    n_jobs=-1
                )
                _p = clean_lgbm_params(_p)
                model = LGBMClassifier(**_p)
            else:
                model = base_model

            # Cross-validated evaluation with early stopping for XGBoost/LightGBM and pruning
            scores = []
            for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                try:
                    if model_name == 'xgboost' and HAS_XGBOOST:
                        fit_kwargs = {'eval_set': [(X_val, y_val)], 'verbose': False, 'early_stopping_rounds': 30}
                        # Optuna pruning callback if available
                        try:
                            import importlib
                            integ = importlib.import_module('optuna.integration')
                            cb = getattr(integ, 'XGBoostPruningCallback', None)
                            if cb is not None:
                                try:
                                    fit_kwargs['callbacks'] = [cb(trial, 'validation_0-logloss')]
                                except Exception as e:
                                    print(f"WARN: Pruning callback integration failed for xgboost: {e}. Continuing without pruning.")
                            else:
                                print("WARN: XGBoostPruningCallback not found in optuna.integration. Training without pruning.")
                        except Exception as e:
                            print(f"WARN: XGBoostPruningCallback import failed: {e}. Training without pruning.")
                        if sample_weight is not None:
                            fit_kwargs['sample_weight'] = sample_weight[train_idx]
                        model.fit(X_tr, y_tr, **fit_kwargs)
                    elif model_name == 'lgbm' and HAS_LIGHTGBM:
                        # Setup eval set and metric; prefer multi_logloss for multiclass
                        n_cls = int(len(np.unique(y_tr)))
                        eval_metric = 'multi_logloss' if n_cls > 2 else 'logloss'
                        fit_kwargs = {'eval_set': [(X_val, y_val)], 'verbose': False, 'eval_metric': eval_metric}
                        # Try to enable Optuna pruning; if unavailable, use robust early stopping
                        pruning_attached = False
                        try:
                            # Try official integration
                            from optuna.integration import LightGBMPruningCallback as _LGBPC  # type: ignore
                            fit_kwargs['callbacks'] = [ _LGBPC(trial, f'valid_0-{eval_metric}') ]
                            pruning_attached = True
                        except Exception:
                            try:
                                # Try legacy optuna_integration package
                                from optuna_integration import LightGBMPruningCallback as _LGBPC  # type: ignore
                                fit_kwargs['callbacks'] = [ _LGBPC(trial, f'valid_0-{eval_metric}') ]
                                pruning_attached = True
                            except Exception as e:
                                print(f"WARN: LightGBMPruningCallback not available ({e}). Using early_stopping_rounds=200 without pruning.")
                        if not pruning_attached:
                            fit_kwargs['early_stopping_rounds'] = 200
                        if sample_weight is not None:
                            fit_kwargs['sample_weight'] = sample_weight[train_idx]
                        model.fit(X_tr, y_tr, **fit_kwargs)
                    else:
                        if sample_weight is not None:
                            model.fit(X_tr, y_tr, sample_weight=sample_weight[train_idx])
                        else:
                            model.fit(X_tr, y_tr)
                except TypeError:
                    if isinstance(model, Pipeline) and sample_weight is not None:
                        model.fit(X_tr, y_tr, **{'clf__sample_weight': sample_weight[train_idx]})
                    else:
                        model.fit(X_tr, y_tr)
                scores.append(auc_score(model, X_val, y_val))
            return float(np.mean(scores))

        study = optuna.create_study(direction='maximize', pruner=pruner)
        print(f"Optuna[{model_name}]: starting optimization (trials={n_trials}, timeout={timeout})")
        # Early stopping for study convergence: stop if no improvement >0.002 over last 10 trials
        last_best = [-np.inf]
        non_improve = [0]
        improve_eps = 0.002
        def _early_stop_cb(study, trial):
            best = study.best_value
            if best is None:
                return
            if best > last_best[0] + improve_eps:
                last_best[0] = best
                non_improve[0] = 0
            else:
                non_improve[0] += 1
                if non_improve[0] >= 10:
                    print("Optuna: stopping early due to plateau (no improvement > 0.002 over last 10 trials)")
                    study.stop()

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True, callbacks=[_early_stop_cb])
        print(f"Optuna[{model_name}]: done. Best value={study.best_value:.4f}")

        best_params = study.best_params
        print(f"Optuna best for {model_name}: {best_params}")
        # Two-stage training: refit with larger n_estimators for the best params
        refit_estimators = None
        if model_name == 'random_forest':
            refit_estimators = 800
        elif model_name == 'xgboost' and HAS_XGBOOST:
            refit_estimators = 600
        elif model_name == 'lgbm' and HAS_LIGHTGBM:
            refit_estimators = 600
        if refit_estimators is not None:
            print(f"Refit best params with large n_estimators: {refit_estimators}")

        # Build final model with best params
        if model_name == 'logistic':
            best = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    random_state=42, max_iter=2000, class_weight='balanced',
                    penalty='l2', solver='lbfgs', C=best_params.get('C', 1.0)
                ))
            ])
        elif model_name == 'random_forest':
            mf_any = best_params.get('max_features', 'sqrt')
            if isinstance(mf_any, (int, float)):
                max_features_rf: float | Literal['sqrt', 'log2'] = float(mf_any)
            elif isinstance(mf_any, str) and mf_any in ('sqrt', 'log2'):
                max_features_rf = cast(Literal['sqrt', 'log2'], mf_any)
            else:
                max_features_rf = 'sqrt'
            best = RandomForestClassifier(
                n_estimators=(refit_estimators if refit_estimators is not None else best_params.get('n_estimators', 100)),
                max_depth=best_params.get('max_depth', None),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                max_features=max_features_rf,
                class_weight='balanced', random_state=42, n_jobs=-1)
        elif model_name == 'xgboost' and HAS_XGBOOST:
            assert xgb is not None
            best = xgb.XGBClassifier(
                random_state=42,
                n_estimators=(refit_estimators if refit_estimators is not None else best_params.get('n_estimators', 100)),
                learning_rate=best_params.get('learning_rate', 0.05),
                max_depth=best_params.get('max_depth', 4),
                min_child_weight=best_params.get('min_child_weight', 3),
                subsample=best_params.get('subsample', 0.8),
                colsample_bytree=best_params.get('colsample_bytree', 0.8),
                gamma=best_params.get('gamma', 0.0),
                reg_alpha=best_params.get('reg_alpha', 0.1),
                reg_lambda=best_params.get('reg_lambda', 1.0),
                eval_metric='logloss')
        elif model_name == 'lgbm' and HAS_LIGHTGBM:
            assert LGBMClassifier is not None
            _p = dict(
                random_state=42,
                n_estimators=(refit_estimators if refit_estimators is not None else best_params.get('n_estimators', 200)),
                learning_rate=best_params.get('learning_rate', 0.05),
                num_leaves=best_params.get('num_leaves', 31),
                feature_fraction=best_params.get('feature_fraction', 0.8),
                bagging_fraction=best_params.get('bagging_fraction', 0.8),
                min_child_samples=best_params.get('min_child_samples', 20),
                reg_alpha=best_params.get('reg_alpha', 0.0),
                reg_lambda=best_params.get('reg_lambda', 1.0),
                class_weight='balanced',
                n_jobs=-1
            )
            _p = clean_lgbm_params(_p)
            best = LGBMClassifier(**_p)
        else:
            best = base_model
        try:
            if sample_weight is not None:
                if isinstance(best, Pipeline):
                    best.fit(X, y, **{'clf__sample_weight': sample_weight})
                else:
                    best.fit(X, y, sample_weight=sample_weight)
            else:
                best.fit(X, y)
        except TypeError:
            if isinstance(best, Pipeline) and sample_weight is not None:
                best.fit(X, y, **{'clf__sample_weight': sample_weight})
            else:
                best.fit(X, y)
        return best

    def train_ensemble(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> VotingClassifier:
        """Train ensemble with hyperparameter tuning; supports sample_weight for imbalance"""
        print("Training individual models...")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

        # Clean and impute features (global)
        X = np.where(np.isfinite(X), X, np.nan)
        # Simple diagnostics for NaN/Inf per column (sample)
        try:
            n_cols = X.shape[1]
            nan_counts = np.sum(np.isnan(X), axis=0)
            inf_counts = np.sum(~np.isfinite(np.where(np.isnan(X), 0.0, X)), axis=0)
            bad_cols = [i for i in range(n_cols) if nan_counts[i] > 0 or inf_counts[i] > 0]
            if bad_cols:
                print(f"Diagnostics: {len(bad_cols)} columns with NaN/Inf before imputation. Example indices: {bad_cols[:10]}")
        except Exception:
            pass
        X_proc = self.imputer.fit_transform(X)

        # Create and tune individual models
        base_models = self.create_models()
        tuned_models = []

        for name, model in base_models.items():
            print(f"Training and tuning {name}...")
            try:
                tuned_model = self.hyperparameter_tuning(X_proc, y, name, model, sample_weight=sample_weight)
                # Probability calibration (isotonic)
                print(f"Calibrating {name}...")
                calibrated_model = CalibratedClassifierCV(tuned_model, method='isotonic', cv=3)
                try:
                    if sample_weight is not None:
                        calibrated_model.fit(X_proc, y, sample_weight=sample_weight)
                    else:
                        calibrated_model.fit(X_proc, y)
                except Exception as e:
                    print(f"WARN: CalibratedClassifierCV with sample_weight failed for {name} ({type(tuned_model).__name__}): {e}. Falling back to unweighted training.")
                    calibrated_model.fit(X_proc, y)
                tuned_models.append((name, calibrated_model))
                self.models[name] = calibrated_model
                print(f"✅ {name} trained and calibrated successfully")
            except Exception as e:
                print(f"❌ {name} training failed: {e}")
                # Fallback: fit default model then calibrate
                try:
                    if sample_weight is not None:
                        model.fit(X_proc, y, sample_weight=sample_weight)
                    else:
                        model.fit(X_proc, y)
                except Exception:
                    model.fit(X_proc, y)
                print(f"Calibrating {name} (fallback)...")
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                try:
                    if sample_weight is not None:
                        calibrated_model.fit(X_proc, y, sample_weight=sample_weight)
                    else:
                        calibrated_model.fit(X_proc, y)
                except Exception as e:
                    print(f"WARN: CalibratedClassifierCV with sample_weight failed for {name} ({type(model).__name__}): {e}. Falling back to unweighted training.")
                    calibrated_model.fit(X_proc, y)
                tuned_models.append((name, calibrated_model))
                self.models[name] = calibrated_model
                print(f"⚠️ {name} using default parameters with calibration")

        if not tuned_models:
            raise RuntimeError("No models could be trained")

        # Create ensemble
        ensemble = VotingClassifier(
            estimators=tuned_models,
            voting='soft'  # Use probabilities
        )

        print("Training ensemble...")
        try:
            if sample_weight is not None:
                ensemble.fit(X_proc, y, sample_weight=sample_weight)
            else:
                ensemble.fit(X_proc, y)
        except TypeError:
            ensemble.fit(X_proc, y)
        print("✅ Ensemble training completed")

        return ensemble

    def train_stacking(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> StackingClassifier:
        """Train stacking ensemble with LR meta-learner; supports sample_weight."""
        print("Training base models for stacking...")
        # Clean and impute features (global)
        X = np.where(np.isfinite(X), X, np.nan)
        try:
            n_cols = X.shape[1]
            nan_counts = np.sum(np.isnan(X), axis=0)
            inf_counts = np.sum(~np.isfinite(np.where(np.isnan(X), 0.0, X)), axis=0)
            bad_cols = [i for i in range(n_cols) if nan_counts[i] > 0 or inf_counts[i] > 0]
            if bad_cols:
                print(f"Diagnostics (stacking): {len(bad_cols)} columns with NaN/Inf before imputation. Example indices: {bad_cols[:10]}")
        except Exception:
            pass
        X_proc = self.imputer.fit_transform(X)
        base_models = self.create_models()
        tuned_models = []
        for name, model in base_models.items():
            print(f"Training and tuning {name} (stacking)...")
            try:
                tuned_model = self.hyperparameter_tuning(X_proc, y, name, model, sample_weight=sample_weight)
                # Probability calibration (isotonic) for stacking base estimator
                print(f"Calibrating {name} (stacking)...")
                calibrated_model = CalibratedClassifierCV(tuned_model, method='isotonic', cv=3)
                try:
                    if sample_weight is not None:
                        calibrated_model.fit(X_proc, y, sample_weight=sample_weight)
                    else:
                        calibrated_model.fit(X_proc, y)
                except Exception as e:
                    print(f"WARN: CalibratedClassifierCV with sample_weight failed for {name} ({type(tuned_model).__name__}): {e}. Falling back to unweighted training.")
                    calibrated_model.fit(X_proc, y)
                self.models[name] = calibrated_model
                tuned_models.append((name, calibrated_model))
            except Exception as e:
                print(f"{name} tuning failed: {e} — using default")
                try:
                    if sample_weight is not None:
                        model.fit(X_proc, y, sample_weight=sample_weight)
                    else:
                        model.fit(X_proc, y)
                except Exception:
                    model.fit(X_proc, y)
                print(f"Calibrating {name} (stacking fallback)...")
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                try:
                    if sample_weight is not None:
                        calibrated_model.fit(X_proc, y, sample_weight=sample_weight)
                    else:
                        calibrated_model.fit(X_proc, y)
                except Exception as e:
                    print(f"WARN: CalibratedClassifierCV with sample_weight failed for {name} ({type(model).__name__}): {e}. Falling back to unweighted training.")
                    calibrated_model.fit(X_proc, y)
                self.models[name] = calibrated_model
                tuned_models.append((name, calibrated_model))
        # Meta-learner with regularization and CV blending
        meta_C = float(getattr(self, 'meta_C', 0.5))
        final_est = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs', penalty='l2', C=meta_C)
        tscv_meta = TimeSeriesSplit(n_splits=int(getattr(self, 'stacking_cv_splits', 3)))
        stack = StackingClassifier(
            estimators=tuned_models,
            final_estimator=final_est,
            passthrough=True,  # include original features
            stack_method='predict_proba',
            cv=tscv_meta
        )
        print("Training stacking classifier (CV blending, passthrough=True)...")
        try:
            if sample_weight is not None:
                stack.fit(X_proc, y, sample_weight=sample_weight)
            else:
                stack.fit(X_proc, y)
        except TypeError:
            stack.fit(X_proc, y)
        print("✅ Stacking training completed")
        return stack

    def train_model(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        if self.model_type == 'stacking':
            return self.train_stacking(X, y, sample_weight=sample_weight)
        elif self.model_type in ('ensemble', 'xgboost', 'random_forest', 'logistic', 'lgbm'):
            return self.train_ensemble(X, y, sample_weight=sample_weight)
        else:
            print(f"Unknown model_type {self.model_type}, defaulting to ensemble")
            return self.train_ensemble(X, y, sample_weight=sample_weight)

    def calculate_feature_importance(self, X: np.ndarray, feature_names: List[str]):
        """Calculate feature importance using multiple methods"""
        importance_dict = {}

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[f'{name}_importance'] = dict(
                    zip(feature_names, model.feature_importances_)
                )
            elif hasattr(model, 'coef_'):
                # For logistic regression, use absolute coefficients
                importance_dict[f'{name}_importance'] = dict(
                    zip(feature_names, np.abs(model.coef_[0]))
                )

        # SHAP values if available
        if HAS_SHAP and 'xgboost' in self.models:
            try:
                assert shap is not None
                explainer = shap.TreeExplainer(self.models['xgboost'])
                shap_values = explainer.shap_values(X[:100])  # Sample for speed
                shap_importance = np.abs(shap_values).mean(0)
                importance_dict['shap_importance'] = dict(
                    zip(feature_names, shap_importance)
                )
            except Exception as e:
                print(f"SHAP calculation failed: {e}")

        self.feature_importance = importance_dict
        return importance_dict

    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                      model_name: str = "model",
                      thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics.
        If `thresholds` is provided and predict_proba is available, use threshold-based
        decisioning instead of plain argmax.
        - Binary: {'binary_threshold': float}
        - Trinary: {'tau': float, 'kappa': float} applied on aggregated DOWN/FLAT/UP probs
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, classification_report, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
        )

        # Preprocess features for evaluation: replace infs, impute
        Xv = np.where(np.isfinite(X), X, np.nan)
        try:
            Xv = self.imputer.transform(Xv)
        except Exception:
            pass

        y_pred_proba = None
        try:
            y_pred_proba = model.predict_proba(Xv)
        except Exception:
            y_pred_proba = None

        # Default predictions via model.predict (argmax inside estimator)
        try:
            y_pred_encoded = model.predict(Xv)
        except Exception:
            y_pred_encoded = None

        # Initialize y_pred to satisfy type checkers; will be overridden below
        y_pred: Any = (y_pred_encoded if y_pred_encoded is not None else np.zeros(Xv.shape[0]))
        used_thresholds = False
        if thresholds is not None and y_pred_proba is not None:
            try:
                # Helper to map proba columns to names
                classes_model = getattr(model, 'classes_', None)
                if classes_model is not None and hasattr(self.label_encoder, 'inverse_transform'):
                    try:
                        class_labels = self.label_encoder.inverse_transform(np.array(classes_model, dtype=int))
                    except Exception:
                        class_labels = np.array(classes_model)
                else:
                    yp = np.asarray(y_pred_proba)
                    n_classes = yp.shape[1] if yp.ndim >= 2 else 2
                    class_labels = np.arange(n_classes)

                def _sanitize(lbl: Any) -> str:
                    s = str(lbl).lower()
                    import re as _re
                    s = _re.sub(r'[^a-z0-9]+', '_', s).strip('_')
                    return f'p_{s}'

                proba_cols = [_sanitize(lbl) for lbl in class_labels]
                import pandas as _pd
                proba_df = _pd.DataFrame(y_pred_proba, columns=proba_cols)

                # Binary thresholding
                if 'binary_threshold' in thresholds and len(proba_cols) == 2:
                    thr = float(thresholds['binary_threshold'])
                    # Use class '1' column if present, else the second column
                    try:
                        idx_pos = int(np.where(np.array(classes_model) == 1)[0][0]) if classes_model is not None else 1
                    except Exception:
                        idx_pos = 1
                    y_pred_bin = (np.asarray(y_pred_proba)[:, idx_pos] >= thr).astype(int)
                    if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
                        try:
                            y_pred = self.label_encoder.inverse_transform(y_pred_bin)
                        except Exception:
                            y_pred = y_pred_bin
                    else:
                        y_pred = y_pred_bin
                    used_thresholds = True
                # Trinary decision rule thresholding
                elif ('tau' in thresholds and 'kappa' in thresholds):
                    try:
                        tri = _canonicalize_tri_proba(proba_df)
                        p_down = tri['p_down']
                        p_up = tri['p_up']
                        tau = float(thresholds['tau'])
                        kappa = float(thresholds['kappa'])
                        max_prob = tri[['p_down', 'p_flat', 'p_up']].max(axis=1)
                        buy = ((p_up - p_down) >= tau) & (max_prob >= kappa)
                        sell = ((p_down - p_up) >= tau) & (max_prob >= kappa)
                        y_pred = np.where(buy, 'UP', np.where(sell, 'DOWN', 'FLAT'))
                        used_thresholds = True
                    except Exception:
                        used_thresholds = False
                else:
                    used_thresholds = False
            except Exception:
                used_thresholds = False

        if not used_thresholds:
            # Fallback to estimator predictions
            if y_pred_encoded is None:
                y_pred = np.zeros(Xv.shape[0])
            else:
                if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
                    try:
                        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
                    except Exception:
                        y_pred = y_pred_encoded
                else:
                    y_pred = y_pred_encoded

        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        balanced_acc = balanced_accuracy_score(y, y_pred)
        try:
            mcc = matthews_corrcoef(y, y_pred)
        except Exception:
            mcc = 0.0

        # Handle multiclass metrics
        avg_method = 'weighted' if len(np.unique(y)) > 2 else 'binary'
        precision = precision_score(y, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y, y_pred, average=avg_method, zero_division=0)
        f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)

        metrics: Dict[str, Any] = {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'mcc': float(mcc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'f1_macro': float(f1_macro),
            'thresholds_applied': bool(used_thresholds),
        }

        # AUC for multiclass (if probabilities available)
        if y_pred_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y)) == 2:
                    auc = roc_auc_score(y, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
                metrics['auc'] = float(auc)
            except Exception as e:
                print(f"Could not calculate AUC: {e}")
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0

        # Brier score (calibration) if probabilities are available
        try:
            from sklearn.metrics import brier_score_loss
            if y_pred_proba is not None and hasattr(model, 'classes_') and hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
                cols_classes = list(getattr(model, 'classes_', []))
                try:
                    col_labels = self.label_encoder.inverse_transform(np.array(cols_classes, dtype=int))
                except Exception:
                    col_labels = np.array(cols_classes)
                briers = []
                for j, lbl in enumerate(col_labels):
                    y_true_bin = (np.array(y) == lbl).astype(int)
                    p = y_pred_proba[:, j]
                    try:
                        b = brier_score_loss(y_true_bin, p)
                        briers.append(b)
                    except Exception:
                        pass
                if briers:
                    metrics['brier_score_mean'] = float(np.mean(briers))
        except Exception:
            pass

        # Print detailed results
        print(f"\n📊 {model_name} Performance:")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Balanced A.:{balanced_acc:.4f}")
        print(f"  MCC:        {metrics.get('mcc', 0.0):.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  F1 (weighted): {f1:.4f}")
        print(f"  F1 (macro):    {metrics.get('f1_macro', 0.0):.4f}")
        print(f"  AUC:        {metrics['auc']:.4f}")
        if 'brier_score_mean' in metrics:
            print(f"  Brier (mean, multiclass one-vs-rest): {metrics['brier_score_mean']:.6f}")

        # Classification report + confusion matrix
        print(f"\n📋 {model_name} Classification Report:")
        try:
            report_text = classification_report(y, y_pred, zero_division=0)
            print(report_text)
            try:
                report_dict_any = classification_report(y, y_pred, output_dict=True, zero_division=0)
                report_dict: Dict[str, Any] = cast(Dict[str, Any], report_dict_any)
                metrics['per_class'] = report_dict
                try:
                    su_any = report_dict.get('STRONG_UP', {})
                    sd_any = report_dict.get('STRONG_DOWN', {})
                    su = cast(Dict[str, Any], su_any) if isinstance(su_any, dict) else {}
                    sd = cast(Dict[str, Any], sd_any) if isinstance(sd_any, dict) else {}
                    metrics['precision_STRONG_UP'] = float(su.get('precision', 0.0))
                    metrics['recall_STRONG_UP'] = float(su.get('recall', 0.0))
                    metrics['precision_STRONG_DOWN'] = float(sd.get('precision', 0.0))
                    metrics['recall_STRONG_DOWN'] = float(sd.get('recall', 0.0))
                    if su or sd:
                        print("Key class metrics:")
                        print(f"  STRONG_UP   -> P: {metrics['precision_STRONG_UP']:.4f}, R: {metrics['recall_STRONG_UP']:.4f}")
                        print(f"  STRONG_DOWN -> P: {metrics['precision_STRONG_DOWN']:.4f}, R: {metrics['recall_STRONG_DOWN']:.4f}")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception as e:
            print(f"Could not generate classification report: {e}")
        try:
            cm = confusion_matrix(y, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
        except Exception as e:
            print(f"Could not compute confusion matrix: {e}")

        return metrics

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[List[Any]] = None,
        tickers: Optional[List[str]] = None,
        test_size: float = 0.2,
        validation: str = 'purged',
        embargo_days: int = 20,
        n_splits: int = 3,
        horizon: int = 20,
        out_dir: Optional[Path] = None,
        ret_fwd: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        dynamic_horizon_k: Optional[float] = None,
        dynamic_embargo: Optional[bool] = None,
        allow_suspicious_features: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train and evaluate with 'simple' or 'purged' validation.
        Returns (model, results) where results contains train/test metrics and importance.
        """
        # Encode labels for estimators
        y_encoded_any = self.label_encoder.fit_transform(y)
        y_encoded = cast(np.ndarray, np.asarray(y_encoded_any))

        # Class imbalance handling
        def make_sample_weight(y_enc: np.ndarray) -> Optional[np.ndarray]:
            try:
                classes = np.unique(y_enc)
                cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_enc)
                class_to_w = {c: w for c, w in zip(classes, cw)}
                return np.asarray([class_to_w[v] for v in y_enc], dtype=float)
            except Exception:
                return None

        # Target alignment audit (feature name hygiene)
        if feature_names is not None:
            if not allow_suspicious_features:
                # Enforce fail-fast on suspicious names
                _check_suspicious_feature_names(list(feature_names))
            else:
                # Allow override for debugging but still warn
                try:
                    _check_suspicious_feature_names(list(feature_names))
                except Exception as e:
                    print(f"WARN: Temporal leakage guard overridden (--allow-suspicious-features): {e}")

        # Simple time split
        if validation != 'purged' or dates is None:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train_enc = cast(np.ndarray, y_encoded[:split_idx])
            y_test_enc = cast(np.ndarray, y_encoded[split_idx:])
            y_train_orig, y_test_orig = y[:split_idx], y[split_idx:]

            sw = make_sample_weight(y_train_enc)
            model = self.train_model(X_train, y_train_enc, sample_weight=sw)

            train_metrics = self.evaluate_model(model, X_train, y_train_orig, "Training Set")
            test_metrics = self.evaluate_model(model, X_test, y_test_orig, "Test Set")

            assert feature_names is not None, "feature_names must be provided from build_dataset"
            fnames = list(feature_names)
            importance = self.calculate_feature_importance(X_train, fnames)

            results: Dict[str, Any] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': importance,
                'feature_names': fnames,
                'model': model,
                'label_encoder': self.label_encoder,
                'validation': 'simple',
            }
            return model, results

        # Purged walk-forward with embargo
        import pandas as _pd
        date_ser = _pd.to_datetime(_pd.Series(dates))
        uniq_dates = sorted(date_ser.dropna().unique())
        if len(uniq_dates) < n_splits + 1:
            n_splits = max(1, min(2, len(uniq_dates) - 1))
        # Survivorship bias mitigation audit (if security_master is available)
        try:
            _try_audit_survivorship(date_ser, tickers)
        except Exception as _e:
            print(f"WARN: Survivorship audit skipped: {_e}")

        date_chunks = np.array_split(np.asarray(uniq_dates), n_splits)
        meta_X_parts: List[np.ndarray] = []
        meta_y_parts: List[int] = []
        # Determine if this task is truly trinary (DOWN/FLAT/UP)
        classes_upper = set(str(c).upper() for c in np.unique(y))
        is_trinary = classes_upper.issuperset({'DOWN', 'FLAT', 'UP'}) and len(classes_upper) == 3
        fold_thresholds: List[Dict[str, Any]] = []


        fold_metrics: List[Dict[str, Any]] = []
        last_model: Any = None
        fold_embargo_bars: List[int] = []

        oof_parts: List[pd.DataFrame] = []
        fold_idx = 0

        for test_dates in date_chunks:
            if len(test_dates) == 0:
                continue
            test_start = test_dates[0]
            test_end = test_dates[-1]
            horizon_days = int(horizon)

            # Define test mask first (used for dynamic embargo computation)
            test_mask = (date_ser >= test_start) & (date_ser <= test_end)

            # Dynamic ATR20-normalized embargo (per-fold 90th percentile of expected bars)
            emb_bars = None
            dyn_enabled = (dynamic_embargo if dynamic_embargo is not None else (dynamic_horizon_k is not None))
            if dyn_enabled:
                try:
                    emb_bars = _compute_dynamic_embargo_bars(
                        feature_names=list(feature_names) if feature_names is not None else None,
                        X=X,
                        mask_test=cast(np.ndarray, np.asarray(test_mask.values, dtype=bool)),
                        horizon=horizon,
                        dynamic_horizon_k=dynamic_horizon_k
                    )
                except Exception:
                    emb_bars = None
            _emb_this = int(emb_bars) if emb_bars is not None else int(embargo_days)
            emb_left = test_start - _pd.Timedelta(days=int(_emb_this + horizon_days))
            # Record per-fold embargo bars actually applied
            try:
                fold_embargo_bars.append(int(_emb_this))
            except Exception:
                pass

            emb_right = test_end + _pd.Timedelta(days=int(_emb_this))
            train_mask = _pd.Series(True, index=date_ser.index)
            tick_ser = _pd.Series(tickers) if tickers is not None and len(tickers) == len(date_ser) else None

            if tick_ser is not None:
                for t in tick_ser[test_mask].unique():
                    t_mask = (tick_ser == t)
                    mask_embargo = t_mask & (date_ser >= emb_left) & (date_ser <= emb_right)
                    train_mask[mask_embargo] = False
            else:
                train_mask = (date_ser < emb_left) | (date_ser > emb_right)

            mask_train = cast(np.ndarray, np.asarray(train_mask.values, dtype=bool))
            mask_test = cast(np.ndarray, np.asarray(test_mask.values, dtype=bool))

            # Strict purged CV overlap check (no train rows inside embargo windows)
            _assert_purged_no_overlap(date_ser, tick_ser, _pd.Series(test_mask), _pd.Series(train_mask), emb_left, emb_right)

            X_train, X_test = X[mask_train], X[mask_test]
            y_train_enc = cast(np.ndarray, y_encoded[mask_train])
            y_test_enc = cast(np.ndarray, y_encoded[mask_test])
            y_test_orig = y[mask_test]
            if len(X_train) < 50 or len(X_test) < 10:
                continue

            sw = make_sample_weight(y_train_enc)
            model_i = self.train_model(X_train, y_train_enc, sample_weight=sw)
            last_model = model_i

            # Compute probabilities for threshold search
            try:
                X_test_v = np.where(np.isfinite(X_test), X_test, np.nan)
                X_test_v = self.imputer.transform(X_test_v)
            except Exception:
                X_test_v = X_test
            y_proba = None
            classes_model = None
            try:
                if hasattr(model_i, 'predict_proba'):
                    y_proba = model_i.predict_proba(X_test_v)
                    classes_model = getattr(model_i, 'classes_', None)
            except Exception:
                y_proba = None

            # Build probability columns matching actual classes if available
            df_fold = None
            proba_cols: List[str] = []
            class_labels: Any = None
            if y_proba is not None:
                if classes_model is not None and hasattr(self.label_encoder, 'inverse_transform'):
                    try:
                        class_labels = self.label_encoder.inverse_transform(np.array(classes_model, dtype=int))
                    except Exception:
                        class_labels = np.array(classes_model)
                else:
                    yp = np.asarray(y_proba)
                    n_classes = yp.shape[1] if yp.ndim >= 2 else (len(getattr(model_i, 'classes_', [])) or 2)
                    class_labels = np.arange(n_classes)
                def _sanitize(lbl: Any) -> str:
                    s = str(lbl).lower()
                    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
                    return f'p_{s}'
                proba_cols = [_sanitize(lbl) for lbl in class_labels]
                df_fold = pd.DataFrame(y_proba, columns=proba_cols)
                df_fold['y_true'] = list(y_test_orig)
                if ret_fwd is not None and len(ret_fwd) == len(y):
                    df_fold['ret_fwd'] = list(ret_fwd[mask_test])
                df_fold['date'] = list(date_ser[mask_test].astype('datetime64[ns]'))
                if tick_ser is not None:
                    df_fold['ticker'] = list(tick_ser[mask_test])
                df_fold['fold'] = int(fold_idx + 1)

            # Per-fold threshold search
            best_thresholds_fold: Optional[Dict[str, Any]] = None
            metric_choice = str(getattr(self, 'threshold_metric', 'balanced_accuracy')).lower()
            try:
                from sklearn.metrics import balanced_accuracy_score as _ba, matthews_corrcoef as _mcc
                if y_proba is not None and is_trinary and df_fold is not None and len(proba_cols) >= 3:
                    tri = _canonicalize_tri_proba(df_fold[proba_cols])
                    y_true_tri = [str(v).upper() for v in list(y_test_orig)]
                    taus = [round(x, 2) for x in np.arange(0.05, 0.55, 0.05)]
                    kappas = [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]
                    best_val = -1e9
                    best_tau, best_kappa = None, None
                    p_down = pd.to_numeric(tri['p_down'], errors='coerce')
                    p_up = pd.to_numeric(tri['p_up'], errors='coerce')
                    max_prob = tri[['p_down', 'p_flat', 'p_up']].max(axis=1)
                    for tau in taus:
                        for kappa in kappas:
                            buy = ((p_up - p_down) >= float(tau)) & (max_prob >= float(kappa))
                            sell = ((p_down - p_up) >= float(tau)) & (max_prob >= float(kappa))
                            y_pred_tri = np.where(buy, 'UP', np.where(sell, 'DOWN', 'FLAT'))
                            if metric_choice == 'mcc':
                                val = float(_mcc(y_true_tri, y_pred_tri))
                            else:
                                val = float(_ba(y_true_tri, y_pred_tri))
                            if val > best_val:
                                best_val, best_tau, best_kappa = val, float(tau), float(kappa)
                    if best_tau is not None and best_kappa is not None:
                        best_thresholds_fold = {'tau': best_tau, 'kappa': best_kappa}
                        fold_thresholds.append({'fold': int(fold_idx + 1), 'tau': best_tau, 'kappa': best_kappa, metric_choice: best_val})
                        print(f"Fold {fold_idx + 1}: best thresholds tau={best_tau:.3f}, kappa={best_kappa:.2f}, {metric_choice}={best_val:.4f}")
                elif y_proba is not None and len(np.unique(y_test_enc)) == 2:
                    thr_grid = [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]
                    try:
                        idx_pos = int(np.where(np.array(classes_model) == 1)[0][0]) if classes_model is not None else 1
                    except Exception:
                        idx_pos = 1
                    best_val = -1e9
                    best_thr = None
                    for t in thr_grid:
                        y_pred_enc = (np.asarray(y_proba)[:, idx_pos] >= t).astype(int)
                        if metric_choice == 'mcc':
                            val = float(_mcc(y_test_enc, y_pred_enc))
                        else:
                            val = float(_ba(y_test_enc, y_pred_enc))
                        if val > best_val:
                            best_val, best_thr = val, float(t)
                    if best_thr is not None:
                        best_thresholds_fold = {'binary_threshold': best_thr}
                        fold_thresholds.append({'fold': int(fold_idx + 1), 'binary_threshold': best_thr, metric_choice: best_val})
                        print(f"Fold {fold_idx + 1}: best binary threshold t={best_thr:.2f}, {metric_choice}={best_val:.4f}")
            except Exception as e:
                print(f"WARN: threshold search skipped on fold {fold_idx + 1}: {e}")

            # Evaluate with thresholds (if any)
            m = self.evaluate_model(model_i, X_test, y_test_orig, "Purged Fold", thresholds=best_thresholds_fold)
            fold_metrics.append(m)

            # Save per-fold OOF and accumulate
            try:
                if df_fold is not None and out_dir is not None:
                    df_fold.to_csv(Path(out_dir) / f'fold_{fold_idx + 1}_oof.csv', index=False)
                if df_fold is not None:
                    oof_parts.append(df_fold)
                # Meta-label dataset
                try:
                    if ret_fwd is not None and len(ret_fwd) == len(y) and y_proba is not None:
                        pred_idx = np.argmax(y_proba, axis=1)
                        pred_labels = [class_labels[int(i)] if int(i) < len(class_labels) else class_labels[0] for i in pred_idx]
                        def _label_dir(lbl: Any) -> int:
                            s = str(lbl).upper()
                            if 'UP' in s:
                                return 1
                            if 'DOWN' in s:
                                return -1
                            return 0
                        dir_pred = np.array([_label_dir(v) for v in pred_labels], dtype=int)
                        r = np.asarray(ret_fwd[mask_test], dtype=float)
                        sign_true = np.sign(r)
                        meta_y = (np.sign(dir_pred) == np.sign(sign_true)).astype(int)
                        meta_X_parts.append(np.asarray(X_test))
                        meta_y_parts.extend(list(meta_y))
                except Exception:
                    pass
            except Exception as e:
                print(f"WARN: OOF collection failed on fold {fold_idx + 1}: {e}")
            finally:
                fold_idx += 1

        def avg_dict(dicts: List[Dict[str, Any]]) -> Dict[str, float]:
            if not dicts:
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc': 0.0}
            keys = set().union(*[d.keys() for d in dicts])
            out: Dict[str, float] = {}
            for k in keys:
                vals = [float(d.get(k, 0.0)) for d in dicts if isinstance(d.get(k, 0.0), (int, float))]
                if vals:
                    out[k] = float(np.mean(vals))
            return out

        test_metrics = avg_dict(fold_metrics)
        importance: Dict[str, Any] = {}
        results: Dict[str, Any] = {
            'train_metrics': {},
            'test_metrics': test_metrics,
            'fold_metrics': fold_metrics,
            'feature_importance': importance,
            'feature_names': (list(feature_names) if feature_names is not None else None),
            'model': last_model,
            'label_encoder': self.label_encoder,
            'validation': 'purged',
            'n_splits': n_splits,
            'embargo_days': embargo_days,
            'embargo_strategy': ('dynamic' if (dynamic_embargo if dynamic_embargo is not None else (dynamic_horizon_k is not None)) else 'static'),
            'fold_embargo_bars': fold_embargo_bars,
        }
        if fold_thresholds:
            try:
                results['fold_thresholds'] = fold_thresholds
                if any('tau' in d for d in fold_thresholds):
                    taus = [d['tau'] for d in fold_thresholds if 'tau' in d]
                    kappas = [d['kappa'] for d in fold_thresholds if 'kappa' in d]
                    if taus and kappas:
                        avg_tau = float(np.mean(taus))
                        avg_kappa = float(np.mean(kappas))
                        results['avg_thresholds'] = {'tau': avg_tau, 'kappa': avg_kappa}
                        print(f"Avg thresholds across folds: tau={avg_tau:.3f}, kappa={avg_kappa:.2f}")
                elif any('binary_threshold' in d for d in fold_thresholds):
                    thrs = [d['binary_threshold'] for d in fold_thresholds if 'binary_threshold' in d]
                    if thrs:
                        avg_thr = float(np.mean(thrs))
                        results['avg_thresholds'] = {'binary_threshold': avg_thr}
                        print(f"Avg binary threshold across folds: t={avg_thr:.3f}")
            except Exception:
                pass

        # Save aggregated OOF and compute adaptive thresholds if available
        if out_dir is not None and oof_parts:
            try:
                oof_df_all = pd.concat(oof_parts, ignore_index=True)
                oof_path = Path(out_dir) / 'oof.csv'
                oof_df_all.to_csv(oof_path, index=False)
                # Compute thresholds only for true trinary labels (DOWN/FLAT/UP)
                try:
                    enc_classes = set(str(c).upper() for c in getattr(self.label_encoder, 'classes_', []))
                    if {'DOWN', 'FLAT', 'UP'}.issubset(enc_classes) and len(enc_classes) == 3:
                        thr = find_thresholds_from_oof(oof_df_all)
                        results['oof_thresholds'] = thr
                        with open(Path(out_dir) / 'oof_thresholds.json', 'w') as f:
                            json.dump(thr, f, indent=2)
                    else:
                        print("INFO: Skipping OOF threshold optimization because labels are not trinary.")
                except Exception as e:
                    print(f"WARN: threshold search on OOF failed: {e}")
            except Exception as e:
                print(f"WARN: could not save aggregated OOF: {e}")

        # Train meta-model (logistic) on OOF-derived meta labels
        try:
            if meta_X_parts and meta_y_parts:
                X_meta = np.vstack(meta_X_parts)
                y_meta = np.asarray(meta_y_parts, dtype=int)
                from sklearn.linear_model import LogisticRegression as _LR
                from sklearn.metrics import accuracy_score as _acc, balanced_accuracy_score as _bacc
                meta_clf = _LR(max_iter=2000, class_weight='balanced', solver='lbfgs')
                # Simple holdout for quick estimate
                split_idx = int(0.8 * len(X_meta))
                Xm_tr, Xm_te = X_meta[:split_idx], X_meta[split_idx:]
                ym_tr, ym_te = y_meta[:split_idx], y_meta[split_idx:]
                meta_clf.fit(Xm_tr, ym_tr)
                ym_pred = meta_clf.predict(Xm_te)
                meta_metrics = {
                    'train_size': int(len(Xm_tr)),
                    'test_size': int(len(Xm_te)),
                    'accuracy': float(_acc(ym_te, ym_pred)),
                    'balanced_accuracy': float(_bacc(ym_te, ym_pred)),
                }
                results['meta'] = meta_metrics
                # Persist meta-model if possible
                try:
                    if out_dir is not None and HAS_JOBLIB:
                        assert joblib is not None
                        meta_path = Path(out_dir) / 'meta_model.joblib'
                        joblib.dump(meta_clf, meta_path)
                        results['meta_model_path'] = str(meta_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"WARN: meta-model training failed: {e}")

        return last_model, results

# --- Systematic Feature Selection (Ablation + optional SHAP) ---

def _group_feature_indices(feature_names: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {
        'trend': [], 'volatility': [], 'volume': [], 'rs': [], 'events': []
    }
    for i, f in enumerate(feature_names):
        fl = f.lower()
        if any(k in fl for k in ['sma', 'ema', 'wma', 'mom', 'momentum', 'roc', 'bb_position', 'kc_position', 'slope', 'trend']):
            groups['trend'].append(i)
        elif any(k in fl for k in ['atr', 'hv', 'volatility', 'bb_width', 'keltner', 'rv_', 'range', 'vix', 'dd_z']):
            groups['volatility'].append(i)
        elif any(k in fl for k in ['volume', 'obv', 'adl', 'dollar_vol', 'vol_']):
            groups['volume'].append(i)
        elif any(k in fl for k in ['rel_strength', 'rs_', 'breadth', 'regime_state']):
            groups['rs'].append(i)
        elif any(k in fl for k in ['fomc', 'cpi', 'ecb', 'nonfarm', 'earnings', 'news', 'event']):
            groups['events'].append(i)
        else:
            # default to trend bucket if unknown
            groups['trend'].append(i)
    return groups


def run_feature_selection(trainer: Any,
                          X: np.ndarray,
                          y: np.ndarray,
                          dates: List[Any],
                          tickers: Optional[List[str]],
                          feature_names: List[str],
                          args: Any) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    groups = _group_feature_indices(feature_names)
    order = ['trend', 'volatility', 'volume', 'rs', 'events']

    def eval_with(idx: List[int]) -> float:
        Xs = X[:, idx]
        _, res = trainer.train_and_evaluate(
            Xs, y, dates=dates, tickers=tickers,
            test_size=0.2, validation=args.validation,
            embargo_days=args.embargo, n_splits=args.n_splits,
            horizon=args.horizon, out_dir=None,
            ret_fwd=None, feature_names=[feature_names[i] for i in idx],
            dynamic_horizon_k=args.dynamic_horizon_k,
            dynamic_embargo=bool(args.dynamic_horizon_k is not None),
            allow_suspicious_features=bool(args.allow_suspicious_features)
        )
        tm = res.get('test_metrics', {})
        return float(tm.get('balanced_accuracy', 0.0))

    selected: List[int] = []
    report: Dict[str, Any] = {'stage1': [], 'stage2': []}

    base_score = None
    # Stage 1: incremental ablation by groups
    for g in order:
        g_idx = groups.get(g, [])
        if not g_idx:
            continue
        cand = sorted(set(selected + g_idx))
        cand_score = eval_with(cand)
        if base_score is None:  # take first available group
            selected = cand
            base_score = cand_score
            report['stage1'].append({'group': g, 'kept': True, 'score': cand_score, 'gain': None})
            continue
        gain = cand_score - base_score
        keep = gain >= float(getattr(args, 'ablation_min_gain', 0.005))
        report['stage1'].append({'group': g, 'kept': bool(keep), 'score': cand_score, 'gain': float(gain)})
        if keep:
            selected = cand
            base_score = cand_score

    # Stage 2: optional SHAP pruning verified via purged OOF
    if getattr(args, 'shap_selection', False) and 'HAS_SHAP' in globals() and HAS_SHAP:
        try:
            assert shap is not None
            if not selected:
                selected = list(range(X.shape[1]))
            # Fit a tree model for SHAP values
            models = trainer.create_models()
            mdl = models.get('xgboost') or models.get('random_forest')
            if mdl is not None:
                from sklearn.preprocessing import LabelEncoder as _LE
                le = _LE()
                y_enc = le.fit_transform(y)
                Xs = X[:, selected]
                mdl.fit(Xs, y_enc)
                # Subsample for speed
                n_sub = min(2000, Xs.shape[0])
                idx_sub = np.linspace(0, Xs.shape[0]-1, n_sub, dtype=int)
                expl = shap.TreeExplainer(mdl)
                sh = expl.shap_values(Xs[idx_sub])
                if isinstance(sh, list):
                    shap_abs = np.mean(np.mean(np.abs(np.array(sh)), axis=0), axis=0)
                else:
                    shap_abs = np.mean(np.abs(sh), axis=0)
                # Rank features by importance ascending
                order_low = np.argsort(shap_abs)
                if base_score is None:
                    base_score = eval_with(selected)

                drops = 0
                for j in order_low[:10]:  # check up to 10 least-informative
                    if drops >= 3:
                        break
                    feat_global_idx = selected[int(j)]
                    trial_idxs = [k for k in selected if k != feat_global_idx]
                    score_minus = eval_with(trial_idxs)
                    # Accept drop if performance doesn't deteriorate more than 0.001
                    if score_minus >= (base_score - 0.001):
                        report['stage2'].append({'dropped_feature': feature_names[feat_global_idx], 'prev_score': base_score, 'new_score': score_minus})
                        selected = trial_idxs
                        base_score = score_minus
                        drops += 1
        except Exception as e:
            print(f"WARN: SHAP selection skipped: {e}")

    X_sel = X[:, selected]
    feat_sel = [feature_names[i] for i in selected]
    report['final'] = {'n_features': len(selected), 'score': base_score}
    return X_sel, feat_sel, report







def load_tickers_from_file(max_tickers: int = 30) -> List[str]:
    """Load tickers from scripts/tickers.txt"""
    tickers_file = REPO_ROOT / 'scripts' / 'tickers.txt'
    if not tickers_file.exists():
        # Fallback to common tickers
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY'][:max_tickers]

    tickers = []
    for line in tickers_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            tickers.append(line.upper())

    return tickers[:max_tickers]

def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic technical features from OHLCV data with improved stability"""
    df = df.copy()

    # SMAs
    df['sma20'] = df['Close'].rolling(window=20).mean()
    df['sma50'] = df['Close'].rolling(window=50).mean()
    df['sma200'] = df['Close'].rolling(window=200).mean()

    # Price ratios (with clipping to prevent extreme values)
    df['price_over_sma20'] = np.clip(df['Close'] / df['sma20'], 0.5, 2.0)
    df['price_over_sma50'] = np.clip(df['Close'] / df['sma50'], 0.5, 2.0)
    df['price_over_sma200'] = np.clip(df['Close'] / df['sma200'], 0.5, 2.0)

    # RSI (classical) and Wilder RSI(2,14)
    delta = pd.to_numeric(df['Close'].diff(), errors='coerce').astype('float64')
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_norm'] = df['rsi'] / 100.0
    # Wilder RSI helper
    def _rsi_wilder(s: pd.Series, n: int) -> pd.Series:
        d = s.diff()
        up = d.clip(lower=0)
        dn = -d.clip(upper=0)
        au = up.ewm(alpha=1.0/n, adjust=False).mean()
        ad = dn.ewm(alpha=1.0/n, adjust=False).mean()
        rs_ = au / (ad + 1e-8)
        return 100 - (100 / (1 + rs_))
    df['rsi2_w'] = _rsi_wilder(df['Close'], 2)
    df['rsi14_w'] = _rsi_wilder(df['Close'], 14)

    # ATR and True Range
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = np.clip(df['atr'] / df['Close'], 0, 0.5)  # Cap at 50%
    # True range ratio vs intraday range
    intraday = (df['High'] - df['Low']).replace(0, np.nan)
    df['tr_ratio'] = (true_range / (intraday + 1e-8)).clip(0, 5)

    # ATR buckets (more stable quantiles)
    atr_pct_clean = df['atr_pct'].dropna()
    if len(atr_pct_clean) > 0:
        atr_quantiles = atr_pct_clean.quantile([0.2, 0.4, 0.6, 0.8])
        df['atr_bucket_0'] = (df['atr_pct'] <= atr_quantiles.iloc[0]).astype(int)
        df['atr_bucket_1'] = ((df['atr_pct'] > atr_quantiles.iloc[0]) &
                             (df['atr_pct'] <= atr_quantiles.iloc[1])).astype(int)
        df['atr_bucket_2'] = ((df['atr_pct'] > atr_quantiles.iloc[1]) &
                             (df['atr_pct'] <= atr_quantiles.iloc[2])).astype(int)
        df['atr_bucket_3'] = ((df['atr_pct'] > atr_quantiles.iloc[2]) &
                             (df['atr_pct'] <= atr_quantiles.iloc[3])).astype(int)
        df['atr_bucket_4'] = (df['atr_pct'] > atr_quantiles.iloc[3]).astype(int)
    else:
        for i in range(5):
            df[f'atr_bucket_{i}'] = 0

    # Volume stats
    vol_ma20 = df['Volume'].rolling(20).mean()
    vol_std20 = df['Volume'].rolling(20).std()
    vol_ma20_lag = vol_ma20.shift(5)
    df['vol20_rising'] = (vol_ma20 > vol_ma20_lag).astype(int)
    df['vol_zscore_20'] = (df['Volume'] - vol_ma20) / (vol_std20 + 1e-8)

    # Momentum/ROC
    df['price_gt_ma20'] = (df['Close'] > df['sma20']).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_momentum'] = np.clip(df['rsi'].diff(5), -50, 50)
    df['roc_5'] = df['Close'].pct_change(5)
    df['roc_10'] = df['Close'].pct_change(10)
    df['roc_20'] = df['Close'].pct_change(20)
    df['momentum_10d'] = df['Close'] - df['Close'].shift(10)
    df['momentum_20d'] = df['Close'] - df['Close'].shift(20)

    # Stochastic oscillators (14,3)
    ll14 = df['Low'].rolling(14).min()
    hh14 = df['High'].rolling(14).max()
    stoch_k = 100 * (df['Close'] - ll14) / (hh14 - ll14 + 1e-8)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_k.rolling(3).mean()

    # SMA slopes over last 10 bars (linear regression slope)
    def _roll_slope(s: pd.Series, w: int) -> pd.Series:
        x = np.arange(w)
        return s.rolling(w).apply(lambda y: np.polyfit(x, y, 1)[0] if np.isfinite(y).all() else np.nan, raw=False)
    df['sma20_slope10'] = _roll_slope(df['sma20'], 10)
    df['sma50_slope10'] = _roll_slope(df['sma50'], 10)
    df['sma200_slope10'] = _roll_slope(df['sma200'], 10)

    # Breakout/Support-Resistance distances
    h20 = df['High'].rolling(20).max(); l20 = df['Low'].rolling(20).min()
    h55 = df['High'].rolling(55).max(); l55 = df['Low'].rolling(55).min()
    df['dist_pct_20d_high'] = (df['Close'] / (h20 + 1e-8)) - 1.0
    df['dist_pct_20d_low'] = (df['Close'] / (l20 + 1e-8)) - 1.0
    df['dist_pct_55d_high'] = (df['Close'] / (h55 + 1e-8)) - 1.0
    df['dist_pct_55d_low'] = (df['Close'] / (l55 + 1e-8)) - 1.0
    near_hi = ((h20 - df['Close']).abs() <= 0.5 * df['atr']).astype(int)
    near_lo = ((df['Close'] - l20).abs() <= 0.5 * df['atr']).astype(int)
    df['break_test_hi_10'] = near_hi.rolling(10).sum()
    df['break_test_lo_10'] = near_lo.rolling(10).sum()
    df['break_test_hi_20'] = near_hi.rolling(20).sum()
    df['break_test_lo_20'] = near_lo.rolling(20).sum()

    # Volatility measures
    close_log_s = pd.to_numeric(df['Close'], errors='coerce').astype('float64').apply(
        lambda v: np.log(v) if (isinstance(v, (float, int)) and v > 0) else np.nan
    )
    log_ret = close_log_s.diff()
    df['hv10'] = log_ret.rolling(10).std()
    df['hv20'] = log_ret.rolling(20).std()

    # Bands: Bollinger and Keltner (EMA20, ATR14, mult=2)
    bb_middle = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    bb_width = bb_upper - bb_lower
    df['bb_position'] = np.clip((df['Close'] - bb_lower) / (bb_width + 1e-8), 0, 1)
    df['bb_width_norm'] = (bb_width / (bb_middle + 1e-8)).replace([np.inf,-np.inf], np.nan)
    df['bb_width_z20'] = (df['bb_width_norm'] - df['bb_width_norm'].rolling(20).mean()) / (df['bb_width_norm'].rolling(20).std() + 1e-8)
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    kc_upper = ema20 + 2.0 * df['atr']
    kc_lower = ema20 - 2.0 * df['atr']
    kc_width = kc_upper - kc_lower
    df['kc_position'] = np.clip((df['Close'] - kc_lower) / (kc_width + 1e-8), 0, 1)
    df['keltner_width_norm'] = (kc_width / (ema20 + 1e-8)).replace([np.inf,-np.inf], np.nan)

    # Volume/Flow: OBV, ADL, up/down volume differential
    chg = df['Close'].diff()
    sign_chg = pd.Series(np.sign(pd.to_numeric(chg, errors='coerce')), index=df.index).fillna(0)
    obv = (sign_chg * df['Volume']).cumsum()
    df['obv'] = obv
    df['obv_slope_10'] = _roll_slope(obv, 10)
    # ADL
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (intraday + 1e-8)
    df['adl'] = (clv * df['Volume']).fillna(0).cumsum()
    # Up/Down volume differential
    up_day = (df['Close'] > df['Close'].shift()).astype(int)
    vol_up_5 = (df['Volume'] * up_day).rolling(5).mean()
    vol_dn_5 = (df['Volume'] * (1 - up_day)).rolling(5).mean()
    df['updown_vol_diff_5'] = (vol_up_5 - vol_dn_5) / (vol_ma20 + 1e-8)
    vol_up_10 = (df['Volume'] * up_day).rolling(10).mean()
    vol_dn_10 = (df['Volume'] * (1 - up_day)).rolling(10).mean()
    df['updown_vol_diff_10'] = (vol_up_10 - vol_dn_10) / (vol_ma20 + 1e-8)

    # Wick ratios
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    df['upper_wick_ratio'] = (df['High'] - df['Close']) / (rng + 1e-8)
    df['lower_wick_ratio'] = (df['Close'] - df['Low']) / (rng + 1e-8)

    # Gap features
    prev_close = df['Close'].shift(1)
    df['gap_pct'] = (df['Open'] - prev_close) / (prev_close + 1e-8)
    gap_filled = ((df['High'] >= prev_close) & (df['Low'] <= prev_close)).astype(float)
    # Approximate gap fill rate over last 5 gaps using 40-bar window
    gap_occ = (df['gap_pct'].abs() > 0).astype(float)
    filled_sum = (gap_filled * gap_occ).rolling(40).sum()
    occ_sum = gap_occ.rolling(40).sum()
    df['gap_fill_rate_5'] = (filled_sum / (occ_sum + 1e-8)).clip(0, 1)

    # Volume ratio (with clipping)
    vol_ratio_raw = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    df['vol_ratio'] = np.clip(vol_ratio_raw, 0.1, 10.0)

    # Price momentum (with clipping)
    df['price_momentum_5d'] = np.clip(df['Close'].pct_change(5), -0.5, 0.5)
    df['price_momentum_10d'] = np.clip(df['Close'].pct_change(10), -0.5, 0.5)

    # Volume trend (with stability)
    vol_trend_raw = df['Volume'].rolling(10).mean() / (df['Volume'].rolling(30).mean() + 1e-8)
    df['volume_trend'] = np.clip(vol_trend_raw, 0.1, 5.0)

    # News sentiment placeholders (TODO: integrate real pipeline e.g., finBERT)
    df['news_sentiment_1d'] = 0.0
    df['news_sentiment_3d'] = 0.0

    return df

def create_labels(df: pd.DataFrame, horizon: int, label_type: str,
                  q_low: float = 0.33, q_high: float = 0.67,
                  dynamic_horizon_k: Optional[float] = None) -> pd.Series:
    """Create labels based on future returns with optional volatility-normalized horizon.
    label_type:
      - 'binary': up/down using fixed threshold (2%)
      - 'multiclass': 5 buckets (legacy)
      - 'trinary': DOWN/FLAT/UP via percentile cutoffs
      - 'regression': raw future return
    If dynamic_horizon_k is provided, compute a per-row horizon (in bars) scaled by ATR20.
    """
    # Compute dynamic per-row horizon in bars if requested
    if dynamic_horizon_k is not None:
        close = pd.to_numeric(df['Close'], errors='coerce').astype('float64')
        # Prefer absolute ATR20 if available; fallback to atr_pct * Close
        if 'atr' in df.columns:
            atr20 = pd.to_numeric(df['atr'], errors='coerce').astype('float64')
        else:
            atr_pct = pd.to_numeric(df.get('atr_pct', pd.Series(0, index=df.index)), errors='coerce').astype('float64')
            atr20 = atr_pct * close
        # Normalize ATR to typical level to avoid extreme horizons; median over series
        atr20_np = atr20.to_numpy(dtype='float64', na_value=np.nan)
        med = np.nanmedian(atr20_np)
        norm = float(med) if np.isfinite(med) else 1.0
        norm = norm if norm > 0 else 1.0
        scale = pd.Series(atr20_np / norm, index=df.index)
        scale = pd.to_numeric(scale, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(1.0)
        # Compute bars per row, clipped to sensible bounds
        bars = (float(horizon) * float(dynamic_horizon_k) * scale).round().astype('int')
        bars = bars.clip(lower=5, upper=60)
        # Forward return over variable horizon
        fr_vals = np.full(len(close), np.nan, dtype='float64')
        cvals = close.values
        bvals = bars.values
        for i in range(len(cvals)):
            j = i + int(bvals[i])
            if j < len(cvals) and np.isfinite(cvals[i]) and np.isfinite(cvals[j]):
                fr_vals[i] = cvals[j] / cvals[i] - 1.0
        future_returns = pd.Series(fr_vals, index=df.index)
    else:
        future_returns = df['Close'].shift(-horizon) / df['Close'] - 1

    fr = pd.to_numeric(future_returns, errors='coerce').astype('float64')

    if label_type == 'binary':
        return (fr > 0.02).astype(int)  # 2% threshold
    elif label_type == 'multiclass':
        labels = pd.Series(index=df.index, dtype='object')
        labels[fr >= 0.10] = 'STRONG_UP'
        labels[(fr >= 0.02) & (fr < 0.10)] = 'WEAK_UP'
        labels[(fr >= -0.02) & (fr < 0.02)] = 'SIDEWAYS'
        labels[(fr >= -0.10) & (fr < -0.02)] = 'WEAK_DOWN'
        labels[fr < -0.10] = 'STRONG_DOWN'
        categories = ['STRONG_DOWN', 'WEAK_DOWN', 'SIDEWAYS', 'WEAK_UP', 'STRONG_UP']
        labels = pd.Categorical(labels, categories=categories, ordered=True)
        return pd.Series(labels, index=df.index)
    elif label_type == 'trinary':
        ret_clean = fr.dropna()
        if len(ret_clean) == 0:
            return pd.Series(index=df.index, dtype='object')
        low_cut = ret_clean.quantile(q_low)
        high_cut = ret_clean.quantile(q_high)
        labels = pd.Series(index=df.index, dtype='object')
        labels[fr <= low_cut] = 'DOWN'
        labels[(fr > low_cut) & (fr < high_cut)] = 'FLAT'
        labels[fr >= high_cut] = 'UP'
        categories = ['DOWN', 'FLAT', 'UP']
        labels = pd.Categorical(labels, categories=categories, ordered=True)
        return pd.Series(labels, index=df.index)
    else:  # regression
        return future_returns

def build_dataset(tickers: List[str], years: int, horizon: int, label_type: str,
                 q_low: float = 0.33, q_high: float = 0.67,
                 dynamic_horizon_k: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str], List[str], np.ndarray]:
    """Build training dataset with universe breadth and optional calendar dummies.
    Returns X, y, dates, tickers (row-wise aligned), feature_names actually used, ret_fwd per row.
    """
    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_dates: List[pd.Timestamp] = []
    all_tickers: List[str] = []
    all_ret_fwd: List[np.ndarray] = []
    final_feature_names: Optional[List[str]] = None

    # 1) Load optional calendars
    fomc_dates: set[pd.Timestamp] = set()


    fomc_expected_change: Dict[pd.Timestamp, float] = {}
    earnings_events: Dict[str, List[Dict[str, Any]]] = {}
    try:
        cal_fomc = Path('config/fomc_dates.csv')
        if cal_fomc.exists():
            df_fomc = pd.read_csv(cal_fomc)
            date_col = 'date' if 'date' in df_fomc.columns else df_fomc.columns[0]
            dates_norm = pd.to_datetime(df_fomc[date_col]).dt.normalize()
            fomc_dates = set(dates_norm)
            # Optional: expected policy change (e.g., bps) or boolean expectation
            if 'expected_change' in df_fomc.columns:
                for d, val in zip(dates_norm, df_fomc['expected_change']):
                    try:
                        fomc_expected_change[d] = float(val)
                    except Exception:
                        pass
            elif 'policy_change' in df_fomc.columns:
                for d, val in zip(dates_norm, df_fomc['policy_change']):
                    try:
                        fomc_expected_change[d] = float(val)
                    except Exception:
                        pass
    except Exception as e:
        print(f"WARN reading FOMC calendar: {e}")
    try:
        cal_earn = Path('config/earnings.csv')
        if cal_earn.exists():
            df_earn = pd.read_csv(cal_earn)
            if 'ticker' in df_earn.columns:
                for t, grp in df_earn.groupby('ticker'):
                    date_col = 'date' if 'date' in grp.columns else grp.columns[-1]
                    events: List[Dict[str, Any]] = []
                    for _, row in grp.iterrows():
                        ev: Dict[str, Any] = {'date': pd.to_datetime(row[date_col]).normalize()}
                        if 'actual' in grp.columns and 'expected' in grp.columns:
                            try:
                                exp = float(row['expected'])
                                act = float(row['actual'])
                                ev['surprise'] = (act - exp) / (abs(exp) + 1e-8)
                            except Exception:
                                ev['surprise'] = 0.0
                        if 'guidance' in grp.columns:
                            try:
                                ev['guidance'] = float(row['guidance'])
                            except Exception:
                                ev['guidance'] = 0.0
                        if 'time' in grp.columns:
                            ev['time'] = str(row['time']).lower()
                        events.append(ev)
                    earnings_events[str(t)] = events
    except Exception as e:
        print(f"WARN reading earnings calendar: {e}")

    # Other macro calendars (optional): CPI, ECB, Non-farm payrolls
    cpi_dates: set[pd.Timestamp] = set()
    ecb_dates: set[pd.Timestamp] = set()
    nonfarm_dates: set[pd.Timestamp] = set()
    def _load_calendar_csv(path: Path, label: str) -> set[pd.Timestamp]:
        try:
            if path.exists():
                dfc = pd.read_csv(path)
                date_col = 'date' if 'date' in dfc.columns else dfc.columns[0]
                return set(pd.to_datetime(dfc[date_col]).dt.normalize())
            else:
                print(f"WARN: calendar file missing: {path}")
                return set()
        except Exception as e:
            print(f"WARN reading {label} calendar: {e}")
            return set()
    cpi_dates = _load_calendar_csv(Path('config/cpi_dates.csv'), 'CPI')
    ecb_dates = _load_calendar_csv(Path('config/ecb_dates.csv'), 'ECB')
    nonfarm_dates = _load_calendar_csv(Path('config/nonfarm_dates.csv'), 'Nonfarm')

    # 2) First pass: fetch and compute per-ticker base features
    data_by_ticker: Dict[str, pd.DataFrame] = {}
    above_map: Dict[str, pd.Series] = {}
    for ticker in tickers:
        print(f"Prefetching {ticker}...")
        df = fetch_history(ticker, years)
        if df is None or len(df) < 100:
            print(f"Skipping {ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
            continue
        feats = compute_basic_features(df)
        try:
            if 'volatility_rank' not in feats.columns:
                vol20 = feats['Close'].pct_change().rolling(20).std()
                feats['volatility_rank'] = vol20.rolling(252).rank(pct=True)
            feats['momentum_diff_5_20'] = (feats['Close'].pct_change(5) - feats['Close'].pct_change(20)).clip(-1, 1)
            feats['liquidity_volume_pct'] = feats['Volume'].rolling(252).rank(pct=True)
            feats['above_sma50_flag'] = (feats['Close'] > feats['sma50']).astype(float)
        except Exception as e:
            print(f"WARN extras for {ticker}: {e}")
        data_by_ticker[ticker] = feats
        above_map[ticker] = feats['above_sma50_flag']

    if not data_by_ticker:
        raise RuntimeError("No valid data found for any ticker")

    # 3) Universe breadth: share of tickers above SMA50 per day
    union_index = None
    for s in above_map.values():
        union_index = s.index if union_index is None else union_index.union(s.index)
    above_df = pd.DataFrame(index=union_index)
    for t, s in above_map.items():
        # Forward-fill only to avoid look-ahead bias (no backfill)
        above_df[t] = s.reindex(union_index).ffill()
    breadth_true = above_df.mean(axis=1).rolling(10, min_periods=1).mean().clip(0, 1)

    # Prioritized features (cap applied later)
    prioritized = [
        # Core trend/vol
        'price_over_sma20','price_over_sma50','price_over_sma200','atr_pct',
        'rsi14_w','rsi2_w','stoch_k','stoch_d',
        'price_momentum_5d','price_momentum_10d','roc_10','momentum_10d',
        'bb_position','kc_position','bb_width_z20','keltner_width_norm',
        'volatility_rank','vol_ratio','vol_zscore_20','sma_alignment','above_all_smas','rsi_momentum',
        'momentum_diff_5_20','liquidity_volume_pct','breadth_proxy',
        # Market context / RS
        'rel_strength_ratio','rel_strength_slope_10','spy_dd_z','spy_range_proxy',
        'breadth_proxy_delta_10d','rv_20_z','rv_10_z','dollar_vol_pct','rv_60_z',
        'regime_state',
        # Volume/flow & microstructure
        'obv_slope_10','adl','upper_wick_ratio','lower_wick_ratio','tr_ratio',
        # Macro calendars (next 3 trading days)
        'is_fomc','days_since_fomc','days_until_fomc','fomc_expected_change','fomc_next_3d',
        'cpi_next_3d','ecb_next_3d','nonfarm_next_3d',
        # Earnings proximity
        'is_earnings_window','days_to_earnings','earnings_surprise','earnings_guidance','earnings_pre','earnings_post',
        # News sentiment (placeholder)
        'news_sentiment_1d','news_sentiment_3d'
    ]

    # Precompute SPY-derived context features
    spy_close = None
    spy_dd_z = None
    spy_range_proxy = None
    try:
        spy_df = fetch_history('SPY', years)
        if spy_df is not None and len(spy_df) >= 100:
            spy_feats = compute_basic_features(spy_df)
            # Core SPY series
            spy_close = pd.to_numeric(spy_feats['Close'], errors='coerce').astype('float64')
            # Drawdown vs 252d rolling high and its z-score
            roll_high = spy_close.rolling(252, min_periods=60).max()
            dd = (spy_close / (roll_high + 1e-8)) - 1.0
            dd_mean = dd.rolling(252, min_periods=60).mean()
            dd_std = dd.rolling(252, min_periods=60).std()
            spy_dd_z = (dd - dd_mean) / (dd_std + 1e-8)
            # VIX proxy via intraday range
            spy_range_proxy = (pd.to_numeric(spy_feats['High'], errors='coerce') - pd.to_numeric(spy_feats['Low'], errors='coerce')) / (spy_close + 1e-8)
    except Exception as e:
        print(f"WARN: could not compute SPY context: {e}")

    # 4) Second pass: align, add breadth + calendar dummies, label, and collect
    for ticker, df_features in data_by_ticker.items():
        print(f"Processing {ticker}...")
        # Breadth as context
        df_features['breadth_proxy'] = breadth_true.reindex(df_features.index).ffill()
        # FOMC features
        idx_norm = pd.to_datetime(df_features.index).normalize()
        if fomc_dates:
            df_features['is_fomc'] = idx_norm.isin(list(fomc_dates)).astype(int)
            # days since/until FOMC
            days_since = []
            days_until = []
            next_exp_change = []
            for d in idx_norm:
                past = [fd for fd in fomc_dates if fd <= d]
                future = [fd for fd in fomc_dates if fd >= d]
                past_date = max(past) if past else None
                future_date = min(future) if future else None
                days_since.append((d - past_date).days if past_date else 999)
                days_until.append((future_date - d).days if future_date else 999)
                next_exp_change.append(float(fomc_expected_change.get(future_date, 0.0)) if future_date else 0.0)
            df_features['days_since_fomc'] = np.array(days_since, dtype=float)
            df_features['days_until_fomc'] = np.array(days_until, dtype=float)
            df_features['fomc_expected_change'] = np.array(next_exp_change, dtype=float)
        else:
            df_features['is_fomc'] = 0
            df_features['days_since_fomc'] = 999.0
            df_features['days_until_fomc'] = 999.0
            df_features['fomc_expected_change'] = 0.0

        # Helper: next 3 trading days indicator for a set of event dates
        def _next3_indicator(index: pd.DatetimeIndex, dates_set: set[pd.Timestamp]) -> np.ndarray:
            if not dates_set or len(index) == 0:
                return np.zeros(len(index), dtype=int)
            arr = np.zeros(len(index), dtype=int)
            idx_list = list(index)
            for i in range(len(idx_list)):
                hit = False
                for j in (1, 2, 3):
                    k = i + j
                    if k < len(idx_list) and idx_list[k].normalize() in dates_set:
                        hit = True
                        break
                arr[i] = 1 if hit else 0
            return arr

        # Macro next-3d dummies
        df_features['fomc_next_3d'] = _next3_indicator(idx_norm, fomc_dates)
        df_features['cpi_next_3d'] = _next3_indicator(idx_norm, cpi_dates)
        df_features['ecb_next_3d'] = _next3_indicator(idx_norm, ecb_dates)
        df_features['nonfarm_next_3d'] = _next3_indicator(idx_norm, nonfarm_dates)

        # Earnings window dummy per ticker (+/- 3 days) and enriched features
        evs = earnings_events.get(ticker, [])
        if evs:
            ewin = pd.Series(0, index=df_features.index, dtype=int)
            surprise = pd.Series(0.0, index=df_features.index, dtype=float)
            guidance = pd.Series(0.0, index=df_features.index, dtype=float)
            is_pre = pd.Series(0, index=df_features.index, dtype=int)
            is_post = pd.Series(0, index=df_features.index, dtype=int)
            # Precompute sorted earnings dates for days_to_earnings calculation
            earn_dates: List[pd.Timestamp] = []
            for _ev in evs:
                _d = _ev.get('date')
                if _d is None:
                    continue
                try:
                    _dd = pd.to_datetime(_d).normalize()
                    if isinstance(_dd, pd.Timestamp):
                        earn_dates.append(_dd)
                except Exception:
                    pass
            earn_dates_sorted = sorted(earn_dates)
            for ev in evs:
                d = ev.get('date')
                if d is None:
                    continue
                left = d - pd.Timedelta(days=3)
                right = d + pd.Timedelta(days=3)
                mask = (df_features.index >= left) & (df_features.index <= right)
                ewin |= mask.astype(int)
                # point features on the event day
                if d in df_features.index:
                    surprise.loc[d] = float(ev.get('surprise', 0.0))
                    guidance.loc[d] = float(ev.get('guidance', 0.0))
                    tval = str(ev.get('time', '')).lower()
                    if tval in ('bmo', 'pre', 'pre-market', 'premarket'):
                        is_pre.loc[d] = 1
                    if tval in ('amc', 'post', 'post-market', 'aftermarket'):
                        is_post.loc[d] = 1
            df_features['is_earnings_window'] = ewin
            # days_to_earnings: min days until next earnings (999 if none)
            dte = []
            for d in df_features.index:
                nxt = [ed for ed in earn_dates_sorted if ed >= d.normalize()]
                dte.append((nxt[0] - d.normalize()).days if nxt else 999)
            df_features['days_to_earnings'] = np.array(dte, dtype=float)
            df_features['earnings_surprise'] = surprise.fillna(0.0)
            df_features['earnings_guidance'] = guidance.fillna(0.0)
            df_features['earnings_pre'] = is_pre
            df_features['earnings_post'] = is_post
        else:
            df_features['is_earnings_window'] = 0
            df_features['days_to_earnings'] = 999.0

            df_features['earnings_surprise'] = 0.0
            df_features['earnings_guidance'] = 0.0
            # Regime classification (60d trend + volatility)
            try:
                trend_60 = df_features['Close'].pct_change(60)
                vol_z_60 = df_features.get('rv_60_z', pd.Series(index=df_features.index, data=np.nan))
                regime = pd.Series(0.0, index=df_features.index)
                regime[(trend_60 > 0.03) & (vol_z_60 < 1.5)] = 1.0
                regime[(trend_60 < -0.03) & (vol_z_60 < 1.5)] = -1.0
                df_features['regime_state'] = regime.fillna(0.0)
            except Exception:
                df_features['regime_state'] = 0.0
            df_features['earnings_pre'] = 0
            df_features['earnings_post'] = 0

        # New regime/signal features (no backfill; only past info)
        try:
            # 10-day change in breadth proxy
            df_features['breadth_proxy_delta_10d'] = df_features['breadth_proxy'] - df_features['breadth_proxy'].shift(10)

            # Realized volatility z-scores
            ret = df_features['Close'].pct_change()

            def _rv_z(ret_ser: pd.Series, w: int) -> pd.Series:
                rv = ret_ser.rolling(window=w, min_periods=max(5, w // 2)).std()
                m = rv.rolling(window=252, min_periods=20).mean()
                s = rv.rolling(window=252, min_periods=20).std()
                return (rv - m) / (s + 1e-8)
            df_features['rv_10_z'] = _rv_z(ret, 10).replace([np.inf, -np.inf], np.nan)
            df_features['rv_20_z'] = _rv_z(ret, 20).replace([np.inf, -np.inf], np.nan)
            df_features['rv_60_z'] = _rv_z(ret, 60).replace([np.inf, -np.inf], np.nan)

            # Dollar volume percentile rank over 252 days
            dv = (df_features['Close'] * df_features['Volume']).astype(float)
            df_features['dollar_vol_pct'] = dv.rolling(window=252, min_periods=20).rank(pct=True)

            # Relative strength vs SPY (Close ratio) and market proxies
            if spy_close is not None:
                spy_aligned = pd.Series(spy_close).reindex(df_features.index).ffill()
                rs = (pd.to_numeric(df_features['Close'], errors='coerce') / (spy_aligned + 1e-8)).replace([np.inf,-np.inf], np.nan)


                df_features['rel_strength_ratio'] = rs
                # 10-bar slope of RS
                x = np.arange(10)
                df_features['rel_strength_slope_10'] = rs.rolling(10).apply(lambda y: np.polyfit(x, y, 1)[0] if np.isfinite(y).all() else np.nan, raw=False)
            if spy_dd_z is not None:
                df_features['spy_dd_z'] = pd.Series(spy_dd_z).reindex(df_features.index).ffill()
            if spy_range_proxy is not None:
                df_features['spy_range_proxy'] = pd.Series(spy_range_proxy).reindex(df_features.index).ffill()
        except Exception as e:
            print(f"WARN: regime features for {ticker} failed: {e}")

        # Labels
        labels = create_labels(df_features, horizon, label_type, q_low=q_low, q_high=q_high, dynamic_horizon_k=dynamic_horizon_k)
        available = [c for c in prioritized if c in df_features.columns]
        if len(available) < 10:
            available = FEATURE_ORDER[:28]
        current_features = list(dict.fromkeys(available))[:28]


        if final_feature_names is None:
            final_feature_names = current_features

        feature_data = df_features[current_features].replace([np.inf, -np.inf], np.nan)
        # Drop initial warm-up rows where rolling windows aren't filled (avoid leakage from back/forward-fill)
        warmup = 252
        feature_data = feature_data.iloc[warmup:]
        label_data = labels.loc[feature_data.index]
        valid_mask = label_data.notna()
        feature_data = feature_data[valid_mask]
        label_data = label_data[valid_mask]
        # Compute aligned forward returns for OOF logging
        future_returns_full = df_features['Close'].shift(-horizon) / df_features['Close'] - 1
        ret_data = future_returns_full.loc[feature_data.index][valid_mask].values

        label_data = label_data[valid_mask]

        if len(feature_data) < 20:
            print(f"Skipping {ticker}: insufficient aligned samples ({len(feature_data)})")
            continue

        all_features.append(np.asarray(feature_data.values))
        all_labels.append(np.asarray(label_data.values))
        all_dates.extend(list(feature_data.index))
        all_tickers.extend([ticker] * len(feature_data))
        all_ret_fwd.append(np.asarray(ret_data))
        print(f"✅ Added {ticker}: {len(feature_data)} samples")

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    ret_fwd = np.concatenate(all_ret_fwd) if all_ret_fwd else np.array([])
    feature_names = final_feature_names if final_feature_names is not None else FEATURE_ORDER[:X.shape[1]]
    return X, y, all_dates, all_tickers, feature_names, ret_fwd

def main():
    parser = argparse.ArgumentParser(description="Train ensemble ML models")
    parser.add_argument('--out-dir', default='ml_out', help='Output directory')
    parser.add_argument('--version', default='v3', help='Model version')
    parser.add_argument('--model-type', default='ensemble',
                       choices=['ensemble', 'stacking', 'xgboost', 'random_forest', 'logistic', 'lgbm'],
                       help='Model type to train')
    parser.add_argument('--years', type=int, default=3, help='Years of data')
    parser.add_argument('--horizon', type=int, default=20, help='Prediction horizon')
    parser.add_argument('--dynamic-horizon-k', type=float, default=None,
                        help='If set, use volatility-normalized horizon per row: bars ~ horizon * k * (ATR20 / median(ATR20))')

    parser.add_argument('--label-type', default='trinary',
                       choices=['binary', 'multiclass', 'trinary', 'regression'],
                       help='Label type')
    parser.add_argument('--q-low', type=float, default=0.33, help='Lower percentile for trinary labels (0-1)')
    parser.add_argument('--q-high', type=float, default=0.67, help='Upper percentile for trinary labels (0-1)')
    parser.add_argument('--validation', default='purged', choices=['simple', 'purged'], help='Validation strategy')
    parser.add_argument('--embargo', type=int, default=20, help='Embargo window in days for purged validation')
    parser.add_argument('--n-splits', type=int, default=3, help='Number of purged folds')
    parser.add_argument('--max-tickers', type=int, default=30, help='Max tickers')

    parser.add_argument('--tuner', default='optuna', choices=['optuna', 'grid'], help='Hyperparameter tuner')
    parser.add_argument('--optuna-trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--optuna-timeout', type=int, default=None, help='Optuna timeout in seconds')
    parser.add_argument('--stacking-cv-splits', type=int, default=3, help='CV splits for stacking meta-features')
    parser.add_argument('--meta-C', type=float, default=0.5, help='Meta-learner (LR) regularization strength C')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow logging if available')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging if available')

    parser.add_argument('--embargo-sensitivity-test', action='store_true',
                        help='Run purged CV with embargo periods [20,30,40] bars and output a comparison report')
    parser.add_argument('--allow-suspicious-features', action='store_true',
                        help='Allow suspicious feature names (temporal leakage guard override)')

    # Threshold optimization and feature selection options
    parser.add_argument('--threshold-metric', default='balanced_accuracy',
                        choices=['balanced_accuracy', 'mcc'],
                        help='Metric to optimize when tuning probability thresholds')
    parser.add_argument('--feature-selection', action='store_true', default=False,
                        help='Enable two-stage feature selection: ablation + SHAP pruning (OOF-validated)')
    parser.add_argument('--ablation-min-gain', type=float, default=0.005,
                        help='Minimum balanced accuracy gain to keep a feature group in ablation (e.g., 0.005 = +0.5pp)')
    parser.add_argument('--shap-selection', action='store_true', default=False,
                        help='Enable SHAP-based redundancy pruning after ablation (each drop OOF-validated)')

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    # Threshold optimization and feature selection options

    out_dir.mkdir(exist_ok=True)

    print(f"Training {args.model_type} model with {args.label_type} labels...")
    print(f"Parameters: years={args.years}, horizon={args.horizon}")

    try:



        # Load tickers and build dataset
        tickers = load_tickers_from_file(args.max_tickers)
        print(f"Loading data for {len(tickers)} tickers...")

        X, y, dates, tickers_row, feature_names, ret_fwd = build_dataset(tickers, args.years, args.horizon, args.label_type,
                                  q_low=args.q_low, q_high=args.q_high, dynamic_horizon_k=args.dynamic_horizon_k)
        print(f"Dataset built: {X.shape[0]} samples, {X.shape[1]} features")

        # Calculate basic stats
        if args.label_type in ('multiclass', 'trinary', 'binary'):
            unique_labels, counts = np.unique(y, return_counts=True)
            label_dist = dict(zip([str(u) for u in unique_labels], counts.tolist()))
        else:
            label_dist = {"mean": float(np.mean(y)), "std": float(np.std(y))}

        # Initialize trainer
        trainer = EnsembleTrainer(args.model_type, tuner=args.tuner)
        # Configure trainer options
        trainer.optuna_trials = int(args.optuna_trials)
        if args.optuna_timeout is not None:
            trainer.optuna_timeout = int(args.optuna_timeout)
        trainer.stacking_cv_splits = int(args.stacking_cv_splits)
        trainer.meta_C = float(args.meta_C)
        trainer.threshold_metric = str(args.threshold_metric)

        # Optional two-stage feature selection (purged OOF validated)
        if bool(args.feature_selection):
            print("\n🧪 Running feature selection (Stage 1: ablation; Stage 2: SHAP, optional)...")
            X_fs, feat_fs, fs_report = run_feature_selection(trainer, X, y, dates, tickers_row, list(feature_names), args)
            # Persist report
            try:
                (out_dir).mkdir(exist_ok=True)
                with open(Path(out_dir) / 'feature_selection.json', 'w') as f:
                    json.dump(fs_report, f, indent=2)
            except Exception as e:
                print(f"WARN: could not persist feature selection report: {e}")
            # Apply selected features to training set
            X = X_fs
            feature_names = feat_fs
            print(f"Selected {len(feature_names)} features after selection.")

        # CI safeguards
        if os.getenv('GITHUB_ACTIONS', '').lower() == 'true':
            # Constrain trials/timeouts to keep runs within GH Actions limits
            trainer.optuna_trials = min(trainer.optuna_trials, 40)
            if getattr(trainer, 'optuna_timeout', None) is None:
                trainer.optuna_timeout = 900  # 15 minutes cap

        # Train and evaluate model
        print(f"\n🚀 Starting {args.model_type} training...")
        model, results = trainer.train_and_evaluate(
            X, y, dates=dates, tickers=tickers_row,
            test_size=0.2, validation=args.validation,
            embargo_days=args.embargo, n_splits=args.n_splits,
            horizon=args.horizon, out_dir=out_dir,
            ret_fwd=ret_fwd, feature_names=feature_names,
            dynamic_horizon_k=args.dynamic_horizon_k,
            dynamic_embargo=bool(args.dynamic_horizon_k is not None),
            allow_suspicious_features=bool(args.allow_suspicious_features)
        )

        # Optional experiment tracking
        try:
            if args.mlflow and HAS_MLFLOW:
                assert mlflow is not None
                mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT', 'stock_dashboard'))
                with mlflow.start_run(run_name=f"{args.model_type}_{args.label_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"):
                    mlflow.log_params({
                        'model_type': args.model_type,
                        'label_type': args.label_type,
                        'years': args.years,
                        'horizon': args.horizon,
                        'tuner': args.tuner,
                        'optuna_trials': int(getattr(trainer, 'optuna_trials', 0)),
                        'optuna_timeout': int(getattr(trainer, 'optuna_timeout', 0) or 0),
                        'validation': args.validation,
                        'embargo_days': args.embargo,
                        'n_splits': args.n_splits
                    })
                    mlflow.log_metrics({
                        'test_accuracy': results['test_metrics'].get('accuracy', 0.0),
                        'test_auc': results['test_metrics'].get('auc', 0.0),
                        'test_f1': results['test_metrics'].get('f1_score', 0.0)
                    })
        except Exception as e:
            print(f"WARN: MLflow logging failed: {e}")

        try:
            if args.wandb and HAS_WANDB:
                assert wandb is not None
                run = wandb.init(project=os.getenv('WANDB_PROJECT', 'stock_dashboard'),
                                 name=f"{args.model_type}_{args.label_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                                 config={
                                     'model_type': args.model_type,
                                     'label_type': args.label_type,
                                     'years': args.years,
                                     'horizon': args.horizon,
                                     'tuner': args.tuner,
                                     'optuna_trials': int(getattr(trainer, 'optuna_trials', 0)),
                                     'optuna_timeout': int(getattr(trainer, 'optuna_timeout', 0) or 0),
                                     'validation': args.validation,
                                     'embargo_days': args.embargo,
                                     'n_splits': args.n_splits
                                 })
                wandb.log({
                    'test/accuracy': results['test_metrics'].get('accuracy', 0.0),
                    'test/auc': results['test_metrics'].get('auc', 0.0),
                    'test/f1': results['test_metrics'].get('f1_score', 0.0)
                })
                wandb.finish()
        except Exception as e:
            print(f"WARN: W&B logging failed: {e}")

        # Persist summary confusion matrix and per-class metrics
        try:
            tm = results.get('test_metrics', {})
            if 'confusion_matrix' in tm:
                np.savetxt(out_dir / 'test_confusion.csv', np.array(tm['confusion_matrix'], dtype=int), delimiter=',', fmt='%d')
            if 'per_class' in tm:
                with open(out_dir / 'test_per_class.json', 'w') as f:
                    json.dump(tm['per_class'], f)
        except Exception as e:
            print(f"WARN: could not save test metrics artifacts: {e}")


        # Create model artifact
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
        # Ensure repo-level models directory exists
        repo_root = Path(__file__).resolve().parents[2]
        models_dir = repo_root / 'ml' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{args.version}_{args.model_type}_{args.label_type}_{timestamp}"
        artifact_pkl_path = models_dir / f"ensemble_{base_name}.pkl"
        # JSON metadata filename in repo models dir
        model_file = models_dir / f"ensemble_{base_name}.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, bool):
                return obj
            elif isinstance(obj, numbers.Integral):
                return int(obj)
            elif isinstance(obj, numbers.Real) and not isinstance(obj, numbers.Integral):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        # Optional: embargo sensitivity testing (static 20/30/40 bar embargo)
        if getattr(args, 'embargo_sensitivity_test', False):
            print("\n🔎 Running embargo sensitivity test for [20, 30, 40] bars...")
            embargo_values = [20, 30, 40]
            run_results: Dict[int, Dict[str, Any]] = {}

            # Local helper for paired permutation p-value (two-sided)
            def _paired_perm_pvalue(a: List[float], b: List[float], n_iter: int = 5000) -> float:
                try:
                    d = np.array(a, dtype=float) - np.array(b, dtype=float)
                    d = d[np.isfinite(d)]
                    if d.size == 0:
                        return 1.0
                    obs = np.mean(d)
                    if np.allclose(d, 0):
                        return 1.0
                    cnt = 0
                    for _ in range(int(max(500, n_iter))):
                        signs = np.random.choice([-1, 1], size=d.size)
                        perm_mean = np.mean(signs * d)
                        if abs(perm_mean) >= abs(obs):
                            cnt += 1
                    return float(cnt) / float(max(1, n_iter))
                except Exception:
                    return 1.0

            for e in embargo_values:
                subdir = out_dir / f"embargo_{e}"
                subdir.mkdir(exist_ok=True)
                _, res = trainer.train_and_evaluate(
                    X, y, dates=dates, tickers=tickers_row,
                    test_size=0.2, validation='purged',
                    embargo_days=int(e), n_splits=int(args.n_splits),
                    horizon=int(args.horizon), out_dir=subdir,
                    ret_fwd=ret_fwd, feature_names=feature_names,
                    dynamic_horizon_k=args.dynamic_horizon_k,
                    dynamic_embargo=False,
                    allow_suspicious_features=bool(args.allow_suspicious_features),
                )
                run_results[int(e)] = res

            # Build comparison report
            metrics_keys = ['accuracy', 'balanced_accuracy', 'f1_score', 'auc']
            report: Dict[str, Any] = {
                'per_embargo': {str(k): run_results[k].get('test_metrics', {}) for k in embargo_values},
                'pairwise': {},
                'n_folds': int(run_results[embargo_values[0]].get('n_splits', args.n_splits)),
            }
            pairs = [(20, 30), (20, 40), (30, 40)]
            for a, b in pairs:
                key = f"{a}_vs_{b}"
                rep_k: Dict[str, Any] = {}
                fa = run_results[a].get('fold_metrics', [])
                fb = run_results[b].get('fold_metrics', [])
                # Align by number of folds
                n = min(len(fa), len(fb))
                for mk in metrics_keys:
                    va = [float(fa[i].get(mk, np.nan)) for i in range(n)]
                    vb = [float(fb[i].get(mk, np.nan)) for i in range(n)]
                    # Drop NaNs pairwise
                    paired = [(x, y) for x, y in zip(va, vb) if np.isfinite(x) and np.isfinite(y)]
                    if paired:
                        xa, xb = zip(*paired)
                        diff = float(np.mean(np.array(xa) - np.array(xb)))
                        pval = _paired_perm_pvalue(list(xa), list(xb), n_iter=3000)
                        rep_k[mk] = {'mean_diff': diff, 'p_value': pval}
                report['pairwise'][key] = rep_k

            # Persist report
            try:
                with open(out_dir / 'embargo_sensitivity.json', 'w') as f:
                    json.dump(report, f, indent=2)
                print("\nEmbargo sensitivity summary (per test metric):")
                for e in embargo_values:
                    tm = report['per_embargo'].get(str(e), {})
                    print(f"  embargo={e}: acc={tm.get('accuracy', 0.):.4f}, ba={tm.get('balanced_accuracy', 0.):.4f}, f1={tm.get('f1_score', 0.):.4f}, auc={tm.get('auc', 0.):.4f}")
                print("\nPairwise significance (mean_diff, p_value):")
                for k, vals in report['pairwise'].items():
                    print(f"  {k}: {vals}")
            except Exception as e:
                print(f"WARN: could not write embargo_sensitivity.json: {e}")


        # Prepare model metadata with real performance metrics
        model_metadata = {
            'version': args.version,
            'model_type': args.model_type,
            'label_type': args.label_type,
            'parameters': {
                'years': args.years,
                'horizon': args.horizon,
                'max_tickers': args.max_tickers,
                'test_size': 0.2,
                'validation': args.validation,
                'embargo_days': args.embargo,
                'n_splits': args.n_splits,
                'q_low': args.q_low,
                'q_high': args.q_high
            },
            'features': feature_names,
            'timestamp': timestamp,
            'dataset_info': {
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'n_tickers': len(tickers),
                'label_distribution': label_dist,
                'train_samples': int(X.shape[0] * 0.8),
                'test_samples': int(X.shape[0] * 0.2)
            },
            'performance': {
                'train': convert_numpy_types(results['train_metrics']),
                'test': convert_numpy_types(results['test_metrics'])
            },
            'feature_importance': convert_numpy_types(results.get('feature_importance', {})),
            'model_info': {
                'ensemble_models': list(trainer.models.keys()),
                'voting_type': 'soft',
                'scaling': 'StandardScaler'
            }
        }

        # Attach threshold info if available
        try:
            if isinstance(results, dict):
                if 'avg_thresholds' in results:
                    model_metadata['avg_thresholds'] = convert_numpy_types(results['avg_thresholds'])
                if 'fold_thresholds' in results:
                    model_metadata['fold_thresholds'] = convert_numpy_types(results['fold_thresholds'])
        except Exception:
            pass

        # Dump pickled model artifact for FastAPI inference
        try:
            if HAS_JOBLIB:
                assert joblib is not None
                artifact = {
                    'model': model,
                    'imputer': getattr(trainer, 'imputer', None),
                    'scaler': getattr(trainer, 'scaler', None),
                    'label_encoder': getattr(trainer, 'label_encoder', None),
                    'feature_names': feature_names,
                    'avg_thresholds': results.get('avg_thresholds'),
                    'timestamp': timestamp,
                    'version': args.version,
                    'model_type': args.model_type,
                    'label_type': args.label_type,
                }
                joblib.dump(artifact, artifact_pkl_path)
            else:
                print("WARN: joblib not available; skipping .pkl artifact dump")
        except Exception as e:
            print(f"WARN: failed to save model artifact: {e}")

        # Save model metadata
        with open(model_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        # Update latest.json
        latest_file = out_dir / 'latest.json'
        latest_data = {
            'json': f'ml/models/{model_file.name}',
            'pkl': f'ml/models/{artifact_pkl_path.name}',
            'version': f'{args.version}_{timestamp}_ensemble'
        }

        with open(latest_file, 'w') as f:
            json.dump(latest_data, f, indent=2)

        # Print summary
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Metadata JSON: {model_file}")
        print(f"📁 Model PKL:    {artifact_pkl_path}")
        print(f"📊 Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"📊 Test AUC: {results['test_metrics']['auc']:.4f}")
        print(f"📊 Test F1: {results['test_metrics']['f1_score']:.4f}")

        # Show top features
        if 'feature_importance' in results and results['feature_importance']:
            print(f"\n🔍 Top 5 Most Important Features:")
            for model_name, importance in results['feature_importance'].items():
                if isinstance(importance, dict):
                    # Convert numpy arrays to scalars for sorting
                    importance_scalars = {}
                    for feat, imp in importance.items():
                        if hasattr(imp, '__len__') and len(imp) > 1:
                            # If it's an array, take the mean
                            importance_scalars[feat] = float(np.mean(imp))
                        else:
                            # If it's a scalar or single-element array
                            importance_scalars[feat] = float(imp)

                    sorted_features = sorted(importance_scalars.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"  {model_name}:")
                    for feat, imp in sorted_features:
                        print(f"    {feat}: {imp:.4f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise



# --- Decision rule and utility metrics helpers ---



def _canonicalize_tri_proba(proba_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns p_down, p_flat, p_up derived from input.
    Supports:
    - trinary input with columns: p_down, p_flat, p_up
    - 5-class input with columns: p_strong_down, p_weak_down, p_sideways, p_weak_up, p_strong_up
    - binary input with columns: p_down, p_up (p_flat inferred)
    Raises ValueError if required columns are missing.
    """
    df = proba_df.copy()
    lower_to_orig = {c.lower(): c for c in df.columns}

    def has_cols(cols: List[str]) -> bool:
        return all(col in lower_to_orig for col in cols)

    # Already trinary
    if has_cols(['p_down', 'p_flat', 'p_up']):
        cols = [lower_to_orig['p_down'], lower_to_orig['p_flat'], lower_to_orig['p_up']]
        return pd.DataFrame({
            'p_down': pd.to_numeric(df[cols[0]], errors='coerce'),
            'p_flat': pd.to_numeric(df[cols[1]], errors='coerce'),
            'p_up': pd.to_numeric(df[cols[2]], errors='coerce'),
        }, index=df.index)

    # 5-class multiclass
    if has_cols(['p_strong_down', 'p_weak_down', 'p_sideways', 'p_weak_up', 'p_strong_up']):
        sd = pd.to_numeric(df[lower_to_orig['p_strong_down']], errors='coerce')
        wd = pd.to_numeric(df[lower_to_orig['p_weak_down']], errors='coerce')
        sw = pd.to_numeric(df[lower_to_orig['p_sideways']], errors='coerce')
        wu = pd.to_numeric(df[lower_to_orig['p_weak_up']], errors='coerce')
        su = pd.to_numeric(df[lower_to_orig['p_strong_up']], errors='coerce')
        return pd.DataFrame({
            'p_down': (sd + wd).clip(lower=0.0, upper=1.0),
            'p_flat': sw.clip(lower=0.0, upper=1.0),
            'p_up': (wu + su).clip(lower=0.0, upper=1.0),
        }, index=df.index)

    # Binary
    if has_cols(['p_down', 'p_up']):
        pdn = pd.to_numeric(df[lower_to_orig['p_down']], errors='coerce')
        pup = pd.to_numeric(df[lower_to_orig['p_up']], errors='coerce')
        pflat = (1.0 - np.maximum(pdn.fillna(0.0), pup.fillna(0.0))).clip(lower=0.0, upper=1.0)
        return pd.DataFrame({'p_down': pdn, 'p_flat': pflat, 'p_up': pup}, index=df.index)

    raise ValueError("Unsupported probability columns. Provide trinary (p_down,p_flat,p_up) or 5-class (p_strong_down,p_weak_down,p_sideways,p_weak_up,p_strong_up) or binary (p_down,p_up).")

def apply_decision_rule(proba_df: pd.DataFrame, tau: float, kappa: float) -> pd.Series:
    """Apply no-trade decision rule on probability DataFrame.
    Works for trinary or 5-class inputs by aggregating to DOWN/FLAT/UP buckets.
    Trade when (p_up - p_down) >= tau AND max_prob >= kappa (BUY),
    or (p_down - p_up) >= tau AND max_prob >= kappa (SELL). Else 0 (no trade).
    Returns a pd.Series with values {+1: BUY, -1: SELL, 0: NO-TRADE}.
    """
    tri = _canonicalize_tri_proba(proba_df)
    p_up = pd.to_numeric(tri['p_up'], errors='coerce')
    p_down = pd.to_numeric(tri['p_down'], errors='coerce')
    max_prob = tri[['p_down', 'p_flat', 'p_up']].max(axis=1)
    buy = ((p_up - p_down) >= float(tau)) & (max_prob >= float(kappa))
    sell = ((p_down - p_up) >= float(tau)) & (max_prob >= float(kappa))
    decisions = np.where(buy, 1, np.where(sell, -1, 0))
    return pd.Series(decisions, index=tri.index, name='decision')


def compute_trade_metrics(proba_df: pd.DataFrame, tau: float, kappa: float, fee_bp: int = 5) -> Dict[str, float]:
    """Compute utility, hit rate, trade rate, and avg return per trade using decision rule.
    Accepts either trinary or 5-class probability columns; aggregates as needed.
    Requires a 'ret_fwd' column for forward returns.
    Utility = mean(trade_ret) - fee_bp/10000.
    """
    df = proba_df.copy()
    if 'ret_fwd' not in df.columns:
        raise ValueError("Missing required column: ret_fwd")
    # Canonicalize to trinary buckets for decision calculation
    tri = _canonicalize_tri_proba(df.drop(columns=[c for c in ['ret_fwd'] if c in df.columns]))
    tri['ret_fwd'] = pd.to_numeric(df['ret_fwd'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    sig = apply_decision_rule(tri[['p_down', 'p_flat', 'p_up']], tau, kappa)
    ret = tri['ret_fwd']
    trade_ret = np.where(sig == 1, ret, np.where(sig == -1, -ret, np.nan))
    trade_ret = pd.Series(trade_ret).dropna()
    n = int(len(tri))
    n_tr = int(len(trade_ret))
    trade_rate = float(n_tr / n) if n > 0 else 0.0
    avg_ret = float(trade_ret.mean()) if n_tr > 0 else 0.0
    hit_rate = float((trade_ret > 0).mean()) if n_tr > 0 else 0.0
    fee = float(fee_bp) / 10000.0
    utility = float(avg_ret - fee)
    return {
        'utility': utility,
        'hit_rate': hit_rate,
        'trade_rate': trade_rate,
        'avg_return_per_trade': avg_ret,
        'n_trades': n_tr,
        'tau': float(tau),
        'kappa': float(kappa),
    }


def find_thresholds_from_oof(oof_df: pd.DataFrame, fee_bp: int = 5) -> Dict[str, float]:
    """Grid search thresholds (tau, kappa) on OOF predictions to maximize utility.
    Accepts trinary or 5-class probability columns and a 'ret_fwd' column.
    Decision rule uses aggregated DOWN/FLAT/UP buckets.
    Returns: { 'tau', 'kappa', 'utility', 'n_trades', 'win_rate', 'trade_rate', 'avg_return_per_trade' }
    """
    df = oof_df.copy()
    if 'ret_fwd' not in df.columns:
        raise ValueError("Missing required OOF column: ret_fwd")
    tri = _canonicalize_tri_proba(df.drop(columns=[c for c in ['ret_fwd'] if c in df.columns]))
    tri['ret_fwd'] = pd.to_numeric(df['ret_fwd'], errors='coerce').replace([np.inf, -np.inf], np.nan)

    taus = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    kappas = [0.5, 0.6, 0.7, 0.8, 0.9]
    best = {'tau': None, 'kappa': None, 'utility': -1e9, 'n_trades': 0, 'win_rate': 0.0, 'trade_rate': 0.0, 'avg_return_per_trade': 0.0}
    fee = float(fee_bp) / 10000.0

    N = int(len(tri))
    for tau in taus:
        for kappa in kappas:
            sig = apply_decision_rule(tri[['p_down', 'p_flat', 'p_up']], tau, kappa)
            trade_ret = np.where(sig == 1, tri['ret_fwd'], np.where(sig == -1, -tri['ret_fwd'], np.nan))
            trade_ret = pd.Series(trade_ret).dropna()
            if len(trade_ret) == 0:
                util = -fee
                n_tr = 0
                win = 0.0
                tr_rate = 0.0
                avg_ret = 0.0
            else:
                avg_ret = float(trade_ret.mean())
                util = float(avg_ret - fee)
                n_tr = int(len(trade_ret))
                win = float((trade_ret > 0).mean())
                tr_rate = float(n_tr / max(1, N))
            if util > best['utility']:
                best = {
                    'tau': float(tau),
                    'kappa': float(kappa),
                    'utility': util,
                    'n_trades': n_tr,
                    'win_rate': win,
                    'trade_rate': tr_rate,
                    'avg_return_per_trade': avg_ret,
                }
    return best

if __name__ == '__main__':
    main()
