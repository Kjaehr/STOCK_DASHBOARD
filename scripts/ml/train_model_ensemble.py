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
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING, cast, Literal
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

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
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

if TYPE_CHECKING:
    from optuna.trial import Trial as OptunaTrial



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
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        # Dynamically configurable attributes (set from main via argparse)
        self.optuna_trials: Optional[int] = None
        self.optuna_timeout: Optional[int] = None
        self.stacking_cv_splits: int = 3
        self.meta_C: float = 0.5


    def create_models(self) -> Dict[str, Any]:
        """Create individual models for ensemble with better regularization"""
        models = {
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',
                C=1.0,  # Regularization strength
                penalty='l2',
                solver='lbfgs'
            ),
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
            models['lgbm'] = LGBMClassifier(
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
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs']
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
            grid_search = GridSearchCV(model, param_grids[model_name],
                                       cv=tscv, scoring='roc_auc_ovr',
                                       n_jobs=-1, verbose=0)
            try:
                if sample_weight is not None:
                    grid_search.fit(X, y, **{'sample_weight': sample_weight})
                else:
                    grid_search.fit(X, y)
            except TypeError:
                grid_search.fit(X, y)
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

        def auc_score(estimator, X_val, y_val):
            try:
                proba = estimator.predict_proba(X_val)
                return float(roc_auc_score(y_val, proba, multi_class='ovr'))
            except Exception:
                preds = estimator.predict(X_val)
                return float(accuracy_score(y_val, preds))

        def objective(trial: 'OptunaTrial') -> float:
            # Expanded search spaces
            if model_name == 'logistic':
                C = trial.suggest_float('C', 1e-4, 1e3, log=True)
                model = LogisticRegression(
                    random_state=42, max_iter=2000, class_weight='balanced',
                    penalty='l2', solver='lbfgs', C=C)
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
                model = LGBMClassifier(
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
                                fit_kwargs['callbacks'] = [cb(trial, 'validation_0-logloss')]
                        except Exception:
                            # XGBoostPruningCallback not available, continue without pruning
                            pass
                        if sample_weight is not None:
                            fit_kwargs['sample_weight'] = sample_weight[train_idx]
                        model.fit(X_tr, y_tr, **fit_kwargs)
                    elif model_name == 'lgbm' and HAS_LIGHTGBM:
                        fit_kwargs = {'eval_set': [(X_val, y_val)], 'verbose': False}
                        # Early stopping for LightGBM
                        try:
                            fit_kwargs['early_stopping_rounds'] = 50
                        except Exception:
                            pass
                        if sample_weight is not None:
                            fit_kwargs['sample_weight'] = sample_weight[train_idx]
                        model.fit(X_tr, y_tr, **fit_kwargs)
                    else:
                        if sample_weight is not None:
                            model.fit(X_tr, y_tr, sample_weight=sample_weight[train_idx])
                        else:
                            model.fit(X_tr, y_tr)
                except TypeError:
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
            best = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced',
                                      penalty='l2', solver='lbfgs', C=best_params.get('C', 1.0))
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
            best = LGBMClassifier(
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
        else:
            best = base_model
        try:
            if sample_weight is not None:
                best.fit(X, y, sample_weight=sample_weight)
            else:
                best.fit(X, y)
        except TypeError:
            best.fit(X, y)
        return best

    def train_ensemble(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> VotingClassifier:
        """Train ensemble with hyperparameter tuning; supports sample_weight for imbalance"""
        print("Training individual models...")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create and tune individual models
        base_models = self.create_models()
        tuned_models = []

        for name, model in base_models.items():
            print(f"Training and tuning {name}...")
            try:
                tuned_model = self.hyperparameter_tuning(X_scaled, y, name, model, sample_weight=sample_weight)
                # Probability calibration (isotonic)
                print(f"Calibrating {name}...")
                calibrated_model = CalibratedClassifierCV(tuned_model, method='isotonic', cv=3)
                try:
                    if sample_weight is not None:
                        calibrated_model.fit(X_scaled, y, sample_weight=sample_weight)
                    else:
                        calibrated_model.fit(X_scaled, y)
                except TypeError:
                    calibrated_model.fit(X_scaled, y)
                tuned_models.append((name, calibrated_model))
                self.models[name] = calibrated_model
                print(f"✅ {name} trained and calibrated successfully")
            except Exception as e:
                print(f"❌ {name} training failed: {e}")
                # Fallback: fit default model then calibrate
                try:
                    if sample_weight is not None:
                        model.fit(X_scaled, y, sample_weight=sample_weight)
                    else:
                        model.fit(X_scaled, y)
                except Exception:
                    model.fit(X_scaled, y)
                print(f"Calibrating {name} (fallback)...")
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                try:
                    if sample_weight is not None:
                        calibrated_model.fit(X_scaled, y, sample_weight=sample_weight)
                    else:
                        calibrated_model.fit(X_scaled, y)
                except TypeError:
                    calibrated_model.fit(X_scaled, y)
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
                ensemble.fit(X_scaled, y, sample_weight=sample_weight)
            else:
                ensemble.fit(X_scaled, y)
        except TypeError:
            ensemble.fit(X_scaled, y)
        print("✅ Ensemble training completed")

        return ensemble

    def train_stacking(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> StackingClassifier:
        """Train stacking ensemble with LR meta-learner; supports sample_weight."""
        print("Training base models for stacking...")
        X_scaled = self.scaler.fit_transform(X)
        base_models = self.create_models()
        tuned_models = []
        for name, model in base_models.items():
            print(f"Training and tuning {name} (stacking)...")
            try:
                tuned_model = self.hyperparameter_tuning(X_scaled, y, name, model, sample_weight=sample_weight)
                # Probability calibration (isotonic) for stacking base estimator
                print(f"Calibrating {name} (stacking)...")
                calibrated_model = CalibratedClassifierCV(tuned_model, method='isotonic', cv=3)
                try:
                    if sample_weight is not None:
                        calibrated_model.fit(X_scaled, y, sample_weight=sample_weight)
                    else:
                        calibrated_model.fit(X_scaled, y)
                except TypeError:
                    calibrated_model.fit(X_scaled, y)
                self.models[name] = calibrated_model
                tuned_models.append((name, calibrated_model))
            except Exception as e:
                print(f"{name} tuning failed: {e} — using default")
                try:
                    if sample_weight is not None:
                        model.fit(X_scaled, y, sample_weight=sample_weight)
                    else:
                        model.fit(X_scaled, y)
                except Exception:
                    model.fit(X_scaled, y)
                print(f"Calibrating {name} (stacking fallback)...")
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                try:
                    if sample_weight is not None:
                        calibrated_model.fit(X_scaled, y, sample_weight=sample_weight)
                    else:
                        calibrated_model.fit(X_scaled, y)
                except TypeError:
                    calibrated_model.fit(X_scaled, y)
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
                stack.fit(X_scaled, y, sample_weight=sample_weight)
            else:
                stack.fit(X_scaled, y)
        except TypeError:
            stack.fit(X_scaled, y)
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
                      model_name: str = "model") -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                   f1_score, classification_report, confusion_matrix)

        # Scale features if needed
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Predictions (model predicts encoded labels)
        y_pred_encoded = model.predict(X_scaled)
        y_pred_proba = None

        try:
            y_pred_proba = model.predict_proba(X_scaled)
        except:
            pass

        # Convert predictions back to original labels (if label encoder exists)
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            try:
                y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
            except Exception:
                y_pred = y_pred_encoded
        else:
            y_pred = y_pred_encoded

        # Basic metrics
        accuracy = accuracy_score(y, y_pred)

        # Handle multiclass metrics
        avg_method = 'weighted' if len(np.unique(y)) > 2 else 'binary'
        precision = precision_score(y, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y, y_pred, average=avg_method, zero_division=0)
        # Macro-averaged F1 for balanced view across classes
        f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)

        metrics: Dict[str, Any] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'f1_macro': float(f1_macro),
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
        except Exception as _e:
            pass

        # Print detailed results
        print(f"\n📊 {model_name} Performance:")
        print(f"  Accuracy:   {accuracy:.4f}")
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
                # Extract STRONG_UP and STRONG_DOWN precision/recall if present
                try:
                    su_any = report_dict.get('STRONG_UP', {})
                    sd_any = report_dict.get('STRONG_DOWN', {})
                    su = cast(Dict[str, Any], su_any) if isinstance(su_any, dict) else {}
                    sd = cast(Dict[str, Any], sd_any) if isinstance(sd_any, dict) else {}
                    metrics['precision_STRONG_UP'] = float(su.get('precision', 0.0))
                    metrics['recall_STRONG_UP'] = float(su.get('recall', 0.0))
                    metrics['precision_STRONG_DOWN'] = float(sd.get('precision', 0.0))
                    metrics['recall_STRONG_DOWN'] = float(sd.get('recall', 0.0))
                    # Console preview
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

            feature_names = FEATURE_ORDER[:X.shape[1]]
            importance = self.calculate_feature_importance(X_train, feature_names)

            results: Dict[str, Any] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': importance,
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
        date_chunks = np.array_split(np.asarray(uniq_dates), n_splits)

        fold_metrics: List[Dict[str, Any]] = []
        last_model: Any = None

        for test_dates in date_chunks:
            if len(test_dates) == 0:
                continue
            test_start = test_dates[0]
            test_end = test_dates[-1]
            horizon_days = int(horizon)
            emb_left = test_start - _pd.Timedelta(days=int(embargo_days + horizon_days))
            emb_right = test_end + _pd.Timedelta(days=int(embargo_days))

            test_mask = (date_ser >= test_start) & (date_ser <= test_end)
            train_mask = _pd.Series(True, index=date_ser.index)
            if tickers is not None and len(tickers) == len(date_ser):
                tick_ser = _pd.Series(tickers)
                for t in tick_ser[test_mask].unique():
                    t_mask = (tick_ser == t)
                    mask_embargo = t_mask & (date_ser >= emb_left) & (date_ser <= emb_right)
                    train_mask[mask_embargo] = False
            else:
                train_mask = (date_ser < emb_left) | (date_ser > emb_right)

            mask_train = cast(np.ndarray, np.asarray(train_mask.values, dtype=bool))
            mask_test = cast(np.ndarray, np.asarray(test_mask.values, dtype=bool))
            X_train, X_test = X[mask_train], X[mask_test]
            y_train_enc = cast(np.ndarray, y_encoded[mask_train])
            y_test_enc = cast(np.ndarray, y_encoded[mask_test])
            y_test_orig = y[mask_test]
            if len(X_train) < 50 or len(X_test) < 10:
                continue

            sw = make_sample_weight(y_train_enc)
            model_i = self.train_model(X_train, y_train_enc, sample_weight=sw)
            last_model = model_i
            m = self.evaluate_model(model_i, X_test, y_test_orig, "Purged Fold")
            fold_metrics.append(m)

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
            'feature_importance': importance,
            'model': last_model,
            'label_encoder': self.label_encoder,
            'validation': 'purged',
            'n_splits': n_splits,
            'embargo_days': embargo_days,
        }
        return last_model, results


        """Train ensemble and evaluate.
        validation: 'simple' (time split) or 'purged' (purged K-fold walk-forward with embargo)
        horizon: prediction horizon in days for horizon-adjusted purging
        """
        # Encode labels to numeric values for model compatibility
        y_encoded = self.label_encoder.fit_transform(y)

        # Helper to build sample weights for imbalance (multiclass-safe via inverse freq)
        def make_sample_weight(y_enc: np.ndarray) -> Optional[np.ndarray]:
            try:
                classes = np.unique(y_enc)
                cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_enc)
                class_to_w = {c: w for c, w in zip(classes, cw)}
                return np.asarray([class_to_w[v] for v in y_enc], dtype=float)
            except Exception:
                return None

        # Simple time split baseline
        if validation != 'purged' or dates is None:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
            y_train_orig, y_test_orig = y[:split_idx], y[split_idx:]

            print(f"Training set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            print(f"Label classes: {list(self.label_encoder.classes_)}")

            sw = make_sample_weight(y_train)
            model = self.train_model(X_train, y_train, sample_weight=sw)

            train_metrics = self.evaluate_model(model, X_train, y_train_orig, "Training Set")
            test_metrics = self.evaluate_model(model, X_test, y_test_orig, "Test Set")

            # Decision-rule metrics on test set (OOF-like)
            try:
                if ret_fwd is not None and len(ret_fwd) == len(y_encoded):
                    X_test_scaled = self.scaler.transform(X_test) if hasattr(self, 'scaler') and self.scaler is not None else X_test
                    proba = None
                    try:
                        proba = model.predict_proba(X_test_scaled)
                    except Exception:
                        proba = None
                    if proba is not None:
                        le_classes = np.array(getattr(self.label_encoder, 'classes_', []))
                        model_classes = np.array(getattr(model, 'classes_', []))
                        def col_for(label: str):
                            if label in le_classes and model_classes.size > 0:
                                enc_val = np.where(le_classes == label)[0]
                                if enc_val.size > 0:
                                    enc_val = int(enc_val[0])
                                    idx = np.where(model_classes == enc_val)[0]
                                    if idx.size > 0:
                                        return int(idx[0])
                            return None
                        idx_down = col_for('DOWN'); idx_flat = col_for('FLAT'); idx_up = col_for('UP')
                        n = len(X_test)
                        p_down = proba[:, idx_down] if idx_down is not None else np.full(n, np.nan)
                        p_flat = proba[:, idx_flat] if idx_flat is not None else np.full(n, np.nan)
                        p_up = proba[:, idx_up] if idx_up is not None else np.full(n, np.nan)
                        ret_test = np.asarray(ret_fwd)[split_idx:]
                        oof_like = pd.DataFrame({'ret_fwd': ret_test, 'p_down': p_down, 'p_flat': p_flat, 'p_up': p_up})
                        thr = find_thresholds_from_oof(oof_like, fee_bp=5)
                        m_rule = compute_trade_metrics(oof_like, thr.get('tau', 0.1), thr.get('kappa', 0.6), fee_bp=5)
                        print(f"Decision-rule Test Utility: {m_rule['utility']:.4f} | Hit Rate: {m_rule['hit_rate']*100:.2f}% | Trade Rate: {m_rule['trade_rate']*100:.2f}% | Avg/Trade: {m_rule['avg_return_per_trade']:.4f}")
                        test_metrics.update({'utility': float(m_rule['utility']), 'hit_rate': float(m_rule['hit_rate']), 'trade_rate': float(m_rule['trade_rate']), 'avg_return_per_trade': float(m_rule['avg_return_per_trade'])})
                        # Duplicate hyphenated keys for JSON compatibility
                        test_metrics['hit-rate'] = float(m_rule['hit_rate'])
                        test_metrics['trade-rate'] = float(m_rule['trade_rate'])

            except Exception as e:
                print(f"WARN: could not compute decision-rule test metrics: {e}")

            # Save per-class metrics for simple validation
            try:
                if out_dir is not None and 'per_class' in test_metrics:
                    (out_dir).mkdir(parents=True, exist_ok=True)
                    with open(out_dir / 'test_per_class.json', 'w') as f:
                        json.dump(test_metrics['per_class'], f)
                    print(f"Saved per-class test metrics to {out_dir / 'test_per_class.json'}")
            except Exception as e:
                print(f"WARN: could not save test_per_class.json (simple): {e}")


            feature_names = FEATURE_ORDER[:X.shape[1]]
            importance = self.calculate_feature_importance(X_train, feature_names)

            results = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': importance,
                'model': model,
                'label_encoder': self.label_encoder
            }
            return model, results

        # Purged walk-forward evaluation with embargo
        import pandas as _pd
        date_ser = _pd.to_datetime(_pd.Series(dates))
        uniq_dates = sorted(date_ser.dropna().unique())
        if len(uniq_dates) < n_splits + 1:
            n_splits = max(1, min(2, len(uniq_dates) - 1))
        date_chunks = np.array_split(np.asarray(uniq_dates), n_splits)


        start_time = time.time()

        fold_metrics: List[Dict[str, float]] = []
        last_model: Any = None
        perm_importances_folds: List[np.ndarray] = []

        for i, test_dates in enumerate(date_chunks):
            if len(test_dates) == 0:
                continue
            test_start = test_dates[0]
            test_end = test_dates[-1]
            horizon_days = int(horizon)
            # Horizon-adjusted purging: extend left embargo by horizon
            emb_left = test_start - _pd.Timedelta(days=int(embargo_days + horizon_days))
            emb_right = test_end + _pd.Timedelta(days=int(embargo_days))

            test_mask = (date_ser >= test_start) & (date_ser <= test_end)
            # Per-ticker embargo purging (group-aware); fallback to global if tickers missing
            train_mask = _pd.Series(True, index=date_ser.index)
            if tickers is not None and len(tickers) == len(date_ser):
                tick_ser = _pd.Series(tickers)
                # For each ticker appearing in test fold, embargo around its test window
                for t in tick_ser[test_mask].unique():
                    t_mask = (tick_ser == t)
                    # Embargo window same for the fold's date span (with horizon-adjusted left)
                    mask_embargo = t_mask & (date_ser >= emb_left) & (date_ser <= emb_right)
                    train_mask[mask_embargo] = False
            else:
                # Global embargo if tickers not provided
                train_mask = (date_ser < emb_left) | (date_ser > emb_right)

            # Validation: ensure no training sample's horizon window overlaps test period
            overlap_mask = (date_ser >= (test_start - _pd.Timedelta(days=horizon_days))) & (date_ser <= test_end)
            violations = int((train_mask & overlap_mask).sum())
            if violations > 0:
                print(f"WARN: Fold {i+1} purging overlap violations removed: {violations}")

            X_train, X_test = X[train_mask.values], X[test_mask.values]
            y_train_enc, y_test_enc = y_encoded[train_mask.values], y_encoded[test_mask.values]
            y_train_orig, y_test_orig = y[train_mask.values], y[test_mask.values]

            if len(X_train) < 50 or len(X_test) < 10:
                print(f"Fold {i+1}: insufficient samples (train {len(X_train)}, test {len(X_test)}); skipping")
                continue

            elapsed = time.time() - start_time
            avg = elapsed / max(1, i)
            eta = avg * (n_splits - i)
            print(f"Fold {i+1}/{n_splits}: train={len(X_train)}, test={len(X_test)} (embargo={embargo_days}d) | ETA ~{eta:.1f}s")
            sw = make_sample_weight(y_train_enc)
            model_i = self.train_model(X_train, y_train_enc, sample_weight=sw)
            last_model = model_i
            # Checkpoint model per fold
            try:
                if out_dir is not None and HAS_JOBLIB:
                    (out_dir).mkdir(parents=True, exist_ok=True)
                    joblib.dump(model_i, out_dir / f"fold_{i+1}_model.pkl")
            except Exception as e:
                print(f"WARN: could not checkpoint fold model: {e}")

            m = self.evaluate_model(model_i, X_test, y_test_orig, f"Purged Fold {i+1}")
            # Save out-of-fold (OOF) predictions for threshold optimization
            try:
                if out_dir is not None:
                    X_test_scaled = self.scaler.transform(X_test) if hasattr(self, 'scaler') and self.scaler is not None else X_test
                    proba = None
                    try:
                        proba = model_i.predict_proba(X_test_scaled)
                    except Exception:
                        proba = None
                    if proba is not None:
                        import numpy as _np
                        # Map probabilities to DOWN/FLAT/UP columns based on label encoder mapping
                        le_classes = _np.array(getattr(self.label_encoder, 'classes_', []))
                        model_classes = _np.array(getattr(model_i, 'classes_', []))
                        def col_for(label: str):
                            if label in le_classes and model_classes.size > 0:
                                enc_val = _np.where(le_classes == label)[0]
                                if enc_val.size > 0:
                                    enc_val = int(enc_val[0])
                                    idx = _np.where(model_classes == enc_val)[0]
                                    if idx.size > 0:
                                        return int(idx[0])
                            return None
                        idx_down = col_for('DOWN')
                        idx_flat = col_for('FLAT')
                        idx_up = col_for('UP')
                        n = len(X_test)
                        p_down = proba[:, idx_down] if idx_down is not None else _np.full(n, _np.nan)
                        p_flat = proba[:, idx_flat] if idx_flat is not None else _np.full(n, _np.nan)
                        p_up = proba[:, idx_up] if idx_up is not None else _np.full(n, _np.nan)
                        dates_fold = _pd.to_datetime(date_ser[test_mask].values)
                        if tickers is not None and len(tickers) == len(date_ser):
                            tickers_fold = _np.asarray(tickers)[test_mask.values]
                        else:
                            tickers_fold = _np.array(['NA'] * n)
                        if ret_fwd is not None and len(ret_fwd) == len(date_ser):
                            ret_fwd_fold = _np.asarray(ret_fwd)[test_mask.values]
                        else:
                            ret_fwd_fold = _np.full(n, _np.nan)
                        oof_df = _pd.DataFrame({
                            'date': dates_fold,
                            'ticker': tickers_fold,
                            'y_true': _np.asarray(y_test_orig),
                            'ret_fwd': ret_fwd_fold,
                            'p_down': p_down,
                            'p_flat': p_flat,
                            'p_up': p_up,
                        })
                        (out_dir).mkdir(parents=True, exist_ok=True)
                        oof_path = out_dir / f"fold_{i+1}_oof.csv"
                        oof_df.to_csv(oof_path, index=False)
                        print(f"Saved OOF predictions to {oof_path}")
            except Exception as e:
                print(f"WARN: could not save OOF predictions for fold {i+1}: {e}")

            # Per-fold permutation importance on validation set
            try:
                from sklearn.inspection import permutation_importance
                # Use the same scaling as predictions
                X_val_scaled = self.scaler.transform(X_test) if hasattr(self, 'scaler') and self.scaler is not None else X_test
                # y for scoring should be encoded to match estimator predictions
                result_pi = permutation_importance(
                    model_i, X_val_scaled, y_test_enc,
                    scoring='f1_weighted', n_repeats=10, random_state=42, n_jobs=-1
                )
                importances_mean = result_pi.importances_mean
                # Save per-fold JSON with feature names and scores
                try:
                    if out_dir is not None:
                        (out_dir).mkdir(parents=True, exist_ok=True)
                        feature_names_fold = FEATURE_ORDER[:X.shape[1]]
                        pi_items = [
                            {'feature': feature_names_fold[j], 'importance': float(importances_mean[j])}
                            for j in range(len(importances_mean))
                        ]
                        with open(out_dir / f"fold_{i+1}_perm_importance.json", 'w') as f:
                            json.dump(pi_items, f)
                        print(f"Saved permutation importance for fold {i+1} -> {out_dir / f'fold_{i+1}_perm_importance.json'}")
                except Exception as e:
                    print(f"WARN: could not save permutation importance for fold {i+1}: {e}")
                # Keep for aggregation
                perm_importances_folds.append(importances_mean)
            except Exception as e:
                print(f"WARN: permutation importance failed for fold {i+1}: {e}")


                # Keep for aggregation
                perm_importances_folds.append(importances_mean)
            except Exception as e:
                print(f"WARN: permutation importance failed for fold {i+1}: {e}")

            fold_metrics.append(m)
            # Save per-fold metrics
            try:
                if out_dir is not None:
                    (out_dir).mkdir(parents=True, exist_ok=True)
                    # Confusion matrix
                    if 'confusion_matrix' in m:
                        cm_path = out_dir / f"fold_{i+1}_confusion.csv"
                        np.savetxt(cm_path, np.array(m['confusion_matrix'], dtype=int), delimiter=",", fmt="%d")
                    # Per-class report
                    if 'per_class' in m:
                        pc_path = out_dir / f"fold_{i+1}_per_class.json"
                        with open(pc_path, 'w') as f:
                            json.dump(m['per_class'], f)
            except Exception as e:
                print(f"WARN: could not save per-fold metrics: {e}")

        # Aggregate metrics across folds
        def avg_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
            if not dicts:
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc': 0.0}
            keys = dicts[0].keys()
            return {k: float(np.mean([d.get(k, 0.0) for d in dicts])) for k in keys}

        test_metrics = avg_dict(fold_metrics)
        importance = {}
        # Aggregate permutation importance across folds
        try:
            if perm_importances_folds:
                agg_arr = np.vstack(perm_importances_folds)
                mean_imp = agg_arr.mean(axis=0)
                std_imp = agg_arr.std(axis=0)
                features_agg = FEATURE_ORDER[:X.shape[1]]
                agg_items = [
                    {'feature': features_agg[j],
                     'importance_mean': float(mean_imp[j]),
                     'importance_std': float(std_imp[j])}
                    for j in range(len(mean_imp))
                ]
                # Save aggregated JSON
                if out_dir is not None:
                    (out_dir).mkdir(parents=True, exist_ok=True)
                    with open(out_dir / 'perm_importance_avg.json', 'w') as f:
                        json.dump(agg_items, f)
                # Console: top-5 features by mean importance
                top_idx = np.argsort(mean_imp)[::-1][:5]
                print("Top-5 permutation importance (averaged across purged folds):")
                for rank, j in enumerate(top_idx, start=1):
                    print(f"  {rank}. {features_agg[j]}: {mean_imp[j]:.6f}")
        except Exception as e:
            print(f"WARN: failed to aggregate permutation importance: {e}")


        # After cross-validation: optional threshold optimization from OOF predictions
        try:
            if out_dir is not None:
                import pandas as _pd
                oof_files = sorted((out_dir).glob("fold_*_oof.csv"))
                if oof_files:
                    oof_list = []
                    for fp in oof_files:
                        try:
                            oof_list.append(_pd.read_csv(fp, parse_dates=["date"]))
                        except Exception as e:
                            print(f"WARN: could not read {fp}: {e}")
                    if oof_list:
                        oof_all = _pd.concat(oof_list, ignore_index=True)
                        thr = find_thresholds_from_oof(oof_all, fee_bp=5)
                        # Save thresholds
                        try:
                            with open(out_dir / "thresholds.json", "w") as f:
                                json.dump(thr, f)
                            print(f"Saved threshold optimization to {out_dir / 'thresholds.json'}")
                        except Exception as e:
                            print(f"WARN: could not save thresholds.json: {e}")

                        # Compute and log OOF decision-rule metrics
                        try:
                            m_rule = compute_trade_metrics(oof_all, thr.get('tau', 0.1), thr.get('kappa', 0.6), fee_bp=5)
                            print(f"Decision-rule OOF Utility: {m_rule['utility']:.4f} | Hit Rate: {m_rule['hit_rate']*100:.2f}% | Trade Rate: {m_rule['trade_rate']*100:.2f}% | Avg/Trade: {m_rule['avg_return_per_trade']:.4f}")
                            test_metrics.update({'utility': float(m_rule['utility']), 'hit_rate': float(m_rule['hit_rate']), 'trade_rate': float(m_rule['trade_rate']), 'avg_return_per_trade': float(m_rule['avg_return_per_trade'])})

                            # Trading summary with optimized thresholds
                            try:
                                decisions = apply_decision_rule(oof_all[['p_down', 'p_flat', 'p_up']], thr.get('tau', 0.1), thr.get('kappa', 0.6))
                                ret = pd.to_numeric(oof_all['ret_fwd'], errors='coerce')
                                trade_ret = pd.Series(np.where(decisions == 1, ret, np.where(decisions == -1, -ret, np.nan))).dropna()
                                total_trades = int(len(trade_ret))
                                win_pct = float((trade_ret > 0).mean()) if total_trades > 0 else 0.0
                                avg_win_return = float(trade_ret[trade_ret > 0].mean()) if (trade_ret > 0).any() else 0.0
                                avg_loss_return = float(trade_ret[trade_ret < 0].mean()) if (trade_ret < 0).any() else 0.0
                                expected_return_per_trade = float(trade_ret.mean()) if total_trades > 0 else 0.0
                                expected_return_total = float(trade_ret.sum()) if total_trades > 0 else 0.0
                                # Update and print
                                test_metrics.update({
                                    'expected_return_per_trade': expected_return_per_trade,
                                    'expected_return_total': expected_return_total,
                                    'total_trades': total_trades,
                                    'win_pct': win_pct,
                                    'avg_win_return': avg_win_return,
                                    'avg_loss_return': avg_loss_return,
                                })
                                print("Trading summary (OOF, optimized thresholds):")
                                print(f"  Trades: {total_trades} | Win %: {win_pct*100:.2f}% | Avg win: {avg_win_return:.5f} | Avg loss: {avg_loss_return:.5f} | Exp/Trade: {expected_return_per_trade:.5f}")
                            except Exception as e:
                                print(f"WARN: could not compute trading summary: {e}")

                            # OOF per-class classification metrics and save
                            try:
                                # Pred label = argmax among DOWN/FLAT/UP
                                prob_mat = np.vstack([oof_all['p_down'].values, oof_all['p_flat'].values, oof_all['p_up'].values]).T
                                idx = np.nanargmax(prob_mat, axis=1)
                                label_map = np.array(['DOWN', 'FLAT', 'UP'])
                                y_pred_oof = label_map[idx]
                                from sklearn.metrics import classification_report
                                report_dict = classification_report(oof_all['y_true'].values, y_pred_oof, output_dict=True, zero_division=0)
                                if out_dir is not None:
                                    (out_dir).mkdir(parents=True, exist_ok=True)
                                    with open(out_dir / 'test_per_class.json', 'w') as f:
                                        json.dump(report_dict, f)
                                    print(f"Saved OOF per-class metrics to {out_dir / 'test_per_class.json'}")
                            except Exception as e:
                                print(f"WARN: could not save OOF per-class metrics: {e}")

                            # Duplicate hyphenated keys for JSON compatibility
                            test_metrics['hit-rate'] = float(m_rule['hit_rate'])
                            test_metrics['trade-rate'] = float(m_rule['trade_rate'])
                        except Exception as e:
                            print(f"WARN: could not compute decision-rule OOF metrics: {e}")
                        except Exception as e:
                            print(f"WARN: could not compute decision-rule OOF metrics: {e}")

        except Exception as e:
            print(f"WARN: threshold optimization failed: {e}")

        results = {
            'train_metrics': {},
            'test_metrics': test_metrics,
            'feature_importance': importance,
            'model': last_model,
            'label_encoder': self.label_encoder,
            'validation': 'purged',
            'n_splits': n_splits,
            'embargo_days': embargo_days
        }
        return last_model, results

        return ensemble, results

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

    # RSI
    delta = pd.to_numeric(df['Close'].diff(), errors='coerce').astype('float64')
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    # Add small epsilon to prevent division by zero
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_norm'] = df['rsi'] / 100.0

    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = np.clip(df['atr'] / df['Close'], 0, 0.5)  # Cap at 50%

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
        # Fallback if no valid data
        for i in range(5):
            df[f'atr_bucket_{i}'] = 0

    # Additional features (with stability improvements)
    vol_ma20 = df['Volume'].rolling(20).mean()
    vol_ma20_lag = vol_ma20.shift(5)
    df['vol20_rising'] = (vol_ma20 > vol_ma20_lag).astype(int)



    df['price_gt_ma20'] = (df['Close'] > df['sma20']).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_momentum'] = np.clip(df['rsi'].diff(5), -50, 50)  # Clip extreme values

    df['sma_alignment'] = ((df['sma20'] > df['sma50']) &
                          (df['sma50'] > df['sma200'])).astype(int)
    df['above_all_smas'] = ((df['Close'] > df['sma20']) &
                           (df['Close'] > df['sma50']) &
                           (df['Close'] > df['sma200'])).astype(int)

    # Volume ratio (with clipping)
    vol_ratio_raw = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    df['vol_ratio'] = np.clip(vol_ratio_raw, 0.1, 10.0)

    # Price momentum (with clipping)
    df['price_momentum_5d'] = np.clip(df['Close'].pct_change(5), -0.5, 0.5)
    df['price_momentum_10d'] = np.clip(df['Close'].pct_change(10), -0.5, 0.5)

    # Bollinger Bands position (with stability)
    bb_middle = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    bb_width = bb_upper - bb_lower
    df['bb_position'] = np.clip((df['Close'] - bb_lower) / (bb_width + 1e-8), 0, 1)

    # Volume trend (with stability)
    vol_trend_raw = df['Volume'].rolling(10).mean() / (df['Volume'].rolling(30).mean() + 1e-8)
    df['volume_trend'] = np.clip(vol_trend_raw, 0.1, 5.0)

    return df

def create_labels(df: pd.DataFrame, horizon: int, label_type: str,
                  q_low: float = 0.33, q_high: float = 0.67) -> pd.Series:
    """Create labels based on future returns.
    label_type:
      - 'binary': up/down using fixed threshold (2%)
      - 'multiclass': 5 buckets (legacy)
      - 'trinary': DOWN/FLAT/UP via percentile cutoffs
      - 'regression': raw future return
    Percentiles computed per-ticker dataframe to be regime-aware.
    """
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
                 q_low: float = 0.33, q_high: float = 0.67) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str], List[str], np.ndarray]:
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

    # Prioritized features (cap to 22 later)
    prioritized = [
        'price_over_sma20','price_over_sma50','price_over_sma200','rsi_norm','atr_pct',
        'price_momentum_5d','price_momentum_10d','bb_position','volume_trend',
        'volatility_rank','vol_ratio','sma_alignment','above_all_smas','rsi_momentum',
        'momentum_diff_5_20','liquidity_volume_pct','breadth_proxy',
        # New regime/signal features
        'rel_strength_spy','breadth_proxy_delta_10d','rv_20_z','rv_10_z','dollar_vol_pct','rv_60_z',
        # Calendar features
        'is_fomc','days_since_fomc','days_until_fomc','fomc_expected_change',
        'is_earnings_window','earnings_surprise','earnings_guidance','earnings_pre','earnings_post'
    ]

    # Precompute SPY relative ratio for rel_strength_spy
    spy_rel = None
    try:
        spy_df = fetch_history('SPY', years)
        if spy_df is not None and len(spy_df) >= 100:
            spy_feats = compute_basic_features(spy_df)
            spy_rel = (spy_feats['Close'] / (spy_feats['sma50'] + 1e-8)).replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        print(f"WARN: could not compute SPY relative series: {e}")

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

        # Earnings window dummy per ticker (+/- 3 days) and enriched features
        evs = earnings_events.get(ticker, [])
        if evs:
            ewin = pd.Series(0, index=df_features.index, dtype=int)
            surprise = pd.Series(0.0, index=df_features.index, dtype=float)
            guidance = pd.Series(0.0, index=df_features.index, dtype=float)
            is_pre = pd.Series(0, index=df_features.index, dtype=int)
            is_post = pd.Series(0, index=df_features.index, dtype=int)
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
            df_features['earnings_surprise'] = surprise.fillna(0.0)
            df_features['earnings_guidance'] = guidance.fillna(0.0)
            df_features['earnings_pre'] = is_pre

            df_features['earnings_post'] = is_post
        else:
            df_features['is_earnings_window'] = 0
            df_features['earnings_surprise'] = 0.0
            df_features['earnings_guidance'] = 0.0
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

            # Relative strength vs SPY (ffill only; no backfill)
            if spy_rel is not None:
                ticker_rel = (df_features['Close'] / (df_features['sma50'] + 1e-8))
                spy_aligned = spy_rel.reindex(df_features.index).ffill()
                df_features['rel_strength_spy'] = (ticker_rel / (spy_aligned + 1e-8)).replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            print(f"WARN: regime features for {ticker} failed: {e}")

        # Labels
        labels = create_labels(df_features, horizon, label_type, q_low=q_low, q_high=q_high)
        available = [c for c in prioritized if c in df_features.columns]
        if len(available) < 10:
            available = FEATURE_ORDER[:22]
        current_features = list(dict.fromkeys(available))[:22]


        if final_feature_names is None:
            final_feature_names = current_features

        feature_data = df_features[current_features]
        feature_data = feature_data.dropna(how='all').ffill().bfill()
        label_data = labels.loc[feature_data.index]
        valid_mask = label_data.notna()
        feature_data = feature_data[valid_mask]
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

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Training {args.model_type} model with {args.label_type} labels...")
    print(f"Parameters: years={args.years}, horizon={args.horizon}")

    try:



        # Load tickers and build dataset
        tickers = load_tickers_from_file(args.max_tickers)
        print(f"Loading data for {len(tickers)} tickers...")

        X, y, dates, tickers_row, feature_names, ret_fwd = build_dataset(tickers, args.years, args.horizon, args.label_type,
                                  q_low=args.q_low, q_high=args.q_high)
        print(f"Dataset built: {X.shape[0]} samples, {X.shape[1]} features")

        # Calculate basic stats
        if args.label_type in ('multiclass', 'trinary', 'binary'):
            unique_labels, counts = np.unique(y, return_counts=True)
            label_dist = dict(zip([str(u) for u in unique_labels], counts.tolist()))
            print(f"Label distribution: {label_dist}")
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
            ret_fwd=ret_fwd
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
        model_file = out_dir / f"model_{args.version}_{timestamp}_ensemble.json"

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

        # Save model metadata
        with open(model_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        # Update latest.json
        latest_file = out_dir / 'latest.json'
        latest_data = {
            'path': f'ml/models/{model_file.name}',
            'version': f'{args.version}_{timestamp}_ensemble'
        }

        with open(latest_file, 'w') as f:
            json.dump(latest_data, f, indent=2)

        # Print summary
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {model_file}")
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

def apply_decision_rule(proba_df: pd.DataFrame, tau: float, kappa: float) -> pd.Series:
    """Apply no-trade decision rule on probability DataFrame.
    Trade when (p_up - p_down) >= tau AND max_prob >= kappa (BUY),
    or (p_down - p_up) >= tau AND max_prob >= kappa (SELL). Else 0 (no trade).
    Returns a pd.Series with values {+1: BUY, -1: SELL, 0: NO-TRADE}.
    Expects columns: p_down, p_flat, p_up
    """
    df = proba_df.copy()
    for c in ['p_down', 'p_flat', 'p_up']:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    p_up = pd.to_numeric(df['p_up'], errors='coerce')
    p_down = pd.to_numeric(df['p_down'], errors='coerce')
    max_prob = df[['p_down', 'p_flat', 'p_up']].max(axis=1)
    buy = ((p_up - p_down) >= float(tau)) & (max_prob >= float(kappa))
    sell = ((p_down - p_up) >= float(tau)) & (max_prob >= float(kappa))
    decisions = np.where(buy, 1, np.where(sell, -1, 0))
    return pd.Series(decisions, index=df.index, name='decision')


def compute_trade_metrics(proba_df: pd.DataFrame, tau: float, kappa: float, fee_bp: int = 5) -> Dict[str, float]:
    """Compute utility, hit rate, trade rate, and avg return per trade using decision rule.
    Expects columns: ret_fwd, p_down, p_flat, p_up.
    Utility is mean(trade_ret) - fee_bp/10000.
    """
    df = proba_df.copy()
    required = ['ret_fwd', 'p_down', 'p_flat', 'p_up']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    # Sanitize numeric columns
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    sig = apply_decision_rule(df[['p_down', 'p_flat', 'p_up']], tau, kappa)
    ret = df['ret_fwd']
    trade_ret = np.where(sig == 1, ret, np.where(sig == -1, -ret, np.nan))
    trade_ret = pd.Series(trade_ret).dropna()
    n = int(len(df))
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
    Decision rule: trade when (p_up - p_down) >= tau AND max_prob >= kappa.
    Expects columns: ret_fwd, p_down, p_flat, p_up
    Utility = mean(trade_ret) - fee_bp/10000, where trade_ret = ret_fwd for BUY, -ret_fwd for SELL.
    Returns: { 'tau', 'kappa', 'utility', 'n_trades', 'win_rate', 'trade_rate', 'avg_return_per_trade' }
    """
    df = oof_df.copy()
    required = ['ret_fwd', 'p_down', 'p_flat', 'p_up']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required OOF column: {c}")
    taus = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    kappas = [0.5, 0.6, 0.7, 0.8, 0.9]
    best = {'tau': None, 'kappa': None, 'utility': -1e9, 'n_trades': 0, 'win_rate': 0.0, 'trade_rate': 0.0, 'avg_return_per_trade': 0.0}
    fee = float(fee_bp) / 10000.0
    # Sanitize
    for col in ['ret_fwd', 'p_down', 'p_flat', 'p_up']:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    N = int(len(df))
    for tau in taus:
        for kappa in kappas:
            # Decisions: +1 buy, -1 sell, 0 no-trade
            sig = apply_decision_rule(df[['p_down', 'p_flat', 'p_up']], tau, kappa)
            trade_ret = np.where(sig == 1, df['ret_fwd'], np.where(sig == -1, -df['ret_fwd'], np.nan))
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
                tr_rate = float(n_tr / N) if N > 0 else 0.0
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
