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
from typing import List, Tuple, Dict, Any, Optional
import json
import numpy as np
import pandas as pd
import requests
import time
import re
from io import StringIO
from urllib.parse import quote_plus
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Install with: pip install shap")

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
    
    def __init__(self, thresholds: Dict[str, float] = None):
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
        
        labels[returns >= self.thresholds['strong_up']] = 'STRONG_UP'
        labels[(returns >= self.thresholds['weak_up']) & 
               (returns < self.thresholds['strong_up'])] = 'WEAK_UP'
        labels[(returns >= -self.thresholds['sideways']) & 
               (returns < self.thresholds['weak_up'])] = 'SIDEWAYS'
        labels[(returns >= -self.thresholds['weak_down']) & 
               (returns < -self.thresholds['sideways'])] = 'WEAK_DOWN'
        labels[returns < -self.thresholds['weak_down']] = 'STRONG_DOWN'
        
        return labels

class EnsembleTrainer:
    """Train ensemble of models with advanced validation"""
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.ensemble = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        
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

        return models
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                            model_name: str, model: Any) -> Any:
        """Perform hyperparameter tuning with time series split"""
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
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name],
                cv=tscv, scoring='roc_auc_ovr',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_
        
        return model
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> VotingClassifier:
        """Train ensemble with hyperparameter tuning"""
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
                tuned_model = self.hyperparameter_tuning(X_scaled, y, name, model)
                tuned_models.append((name, tuned_model))
                self.models[name] = tuned_model
                print(f"✅ {name} trained successfully")
            except Exception as e:
                print(f"❌ {name} training failed: {e}")
                # Add untrained model as fallback
                model.fit(X_scaled, y)
                tuned_models.append((name, model))
                self.models[name] = model
                print(f"⚠️ {name} using default parameters")

        if not tuned_models:
            raise RuntimeError("No models could be trained")

        # Create ensemble
        ensemble = VotingClassifier(
            estimators=tuned_models,
            voting='soft'  # Use probabilities
        )

        print("Training ensemble...")
        ensemble.fit(X_scaled, y)
        print("✅ Ensemble training completed")

        return ensemble
    
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
                      model_name: str = "model") -> Dict[str, float]:
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

        # Convert predictions back to original labels if needed
        if hasattr(self, 'label_encoder') and hasattr(y[0], '__class__') and isinstance(y[0], str):
            # y is original string labels, convert predictions back
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            # y is already encoded, use encoded predictions
            y_pred = y_pred_encoded

        # Basic metrics
        accuracy = accuracy_score(y, y_pred)

        # Handle multiclass metrics
        avg_method = 'weighted' if len(np.unique(y)) > 2 else 'binary'
        precision = precision_score(y, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y, y_pred, average=avg_method, zero_division=0)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
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

        # Print detailed results
        print(f"\n📊 {model_name} Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")

        # Classification report
        print(f"\n📋 {model_name} Classification Report:")
        try:
            report = classification_report(y, y_pred, zero_division=0)
            print(report)
        except Exception as e:
            print(f"Could not generate classification report: {e}")

        return metrics

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray,
                          test_size: float = 0.2) -> Tuple[Any, Dict[str, Any]]:
        """Train ensemble and evaluate with train/test split"""
        from sklearn.model_selection import train_test_split

        # Encode labels to numeric values for XGBoost compatibility
        y_encoded = self.label_encoder.fit_transform(y)

        # Time-aware split (important for financial data)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
        y_train_orig, y_test_orig = y[:split_idx], y[split_idx:]

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Label classes: {list(self.label_encoder.classes_)}")

        # Train ensemble
        ensemble = self.train_ensemble(X_train, y_train)

        # Evaluate on training set (convert back to original labels for metrics)
        train_metrics = self.evaluate_model(ensemble, X_train, y_train_orig, "Training Set")

        # Evaluate on test set (convert back to original labels for metrics)
        test_metrics = self.evaluate_model(ensemble, X_test, y_test_orig, "Test Set")

        # Calculate feature importance
        feature_names = FEATURE_ORDER[:X.shape[1]]
        importance = self.calculate_feature_importance(X_train, feature_names)

        # Combine results
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': importance,
            'model': ensemble,
            'label_encoder': self.label_encoder
        }

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
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Add small epsilon to prevent division by zero
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_norm'] = df['rsi'] / 100.0

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
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

def create_labels(df: pd.DataFrame, horizon: int, label_type: str) -> pd.Series:
    """Create labels based on future returns"""
    future_returns = df['Close'].shift(-horizon) / df['Close'] - 1

    if label_type == 'binary':
        return (future_returns > 0.02).astype(int)  # 2% threshold
    elif label_type == 'multiclass':
        # Create labels as strings first, then convert to categorical
        labels = pd.Series(index=df.index, dtype='object')
        labels[future_returns >= 0.10] = 'STRONG_UP'
        labels[(future_returns >= 0.02) & (future_returns < 0.10)] = 'WEAK_UP'
        labels[(future_returns >= -0.02) & (future_returns < 0.02)] = 'SIDEWAYS'
        labels[(future_returns >= -0.10) & (future_returns < -0.02)] = 'WEAK_DOWN'
        labels[future_returns < -0.10] = 'STRONG_DOWN'

        # Convert to categorical with predefined categories
        categories = ['STRONG_DOWN', 'WEAK_DOWN', 'SIDEWAYS', 'WEAK_UP', 'STRONG_UP']
        labels = pd.Categorical(labels, categories=categories, ordered=True)
        return pd.Series(labels, index=df.index)
    else:  # regression
        return future_returns

def build_dataset(tickers: List[str], years: int, horizon: int, label_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build training dataset from multiple tickers"""
    all_features = []
    all_labels = []

    # Use only the first 22 features that we can actually compute
    current_features = FEATURE_ORDER[:22]

    for ticker in tickers:
        print(f"Processing {ticker}...")
        df = fetch_history(ticker, years)
        if df is None or len(df) < 100:  # Reduced minimum data requirement
            print(f"Skipping {ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
            continue

        print(f"  Raw data: {len(df)} rows")

        # Compute features
        df_features = compute_basic_features(df)
        print(f"  After features: {len(df_features)} rows")

        # Create labels
        labels = create_labels(df_features, horizon, label_type)
        print(f"  Labels created: {len(labels)} rows, {labels.notna().sum()} non-null")

        # Extract feature matrix (be more lenient with NaN handling)
        feature_data = df_features[current_features]
        print(f"  Feature data shape: {feature_data.shape}")
        print(f"  Feature data non-null: {feature_data.notna().all(axis=1).sum()} complete rows")

        # Only drop rows where ALL features are NaN
        feature_data = feature_data.dropna(how='all')

        # For remaining NaN values, forward fill then backward fill
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')

        # Get labels for the same index
        label_data = labels.loc[feature_data.index]

        # Remove rows where labels are NaN
        valid_mask = label_data.notna()
        feature_data = feature_data[valid_mask]
        label_data = label_data[valid_mask]

        print(f"  Final aligned data: {len(feature_data)} samples")

        if len(feature_data) < 20:  # Reduced minimum samples
            print(f"Skipping {ticker}: insufficient aligned samples ({len(feature_data)} < 20)")
            continue

        feature_matrix = feature_data.values
        label_vector = label_data.values

        all_features.append(feature_matrix)
        all_labels.append(label_vector)

        print(f"✅ Added {ticker}: {len(feature_data)} samples")

    if not all_features:
        raise RuntimeError("No valid data found for any ticker")

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train ensemble ML models")
    parser.add_argument('--out-dir', default='ml_out', help='Output directory')
    parser.add_argument('--version', default='v3', help='Model version')
    parser.add_argument('--model-type', default='ensemble',
                       choices=['ensemble', 'xgboost', 'random_forest', 'logistic'],
                       help='Model type to train')
    parser.add_argument('--years', type=int, default=3, help='Years of data')
    parser.add_argument('--horizon', type=int, default=20, help='Prediction horizon')
    parser.add_argument('--label-type', default='multiclass',
                       choices=['binary', 'multiclass', 'regression'],
                       help='Label type')
    parser.add_argument('--max-tickers', type=int, default=30, help='Max tickers')

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

        X, y = build_dataset(tickers, args.years, args.horizon, args.label_type)
        print(f"Dataset built: {X.shape[0]} samples, {X.shape[1]} features")

        # Calculate basic stats
        if args.label_type == 'multiclass':
            unique_labels, counts = np.unique(y, return_counts=True)
            label_dist = dict(zip(unique_labels, counts.tolist()))
            print(f"Label distribution: {label_dist}")
        else:
            label_dist = {"mean": float(np.mean(y)), "std": float(np.std(y))}

        # Initialize trainer
        trainer = EnsembleTrainer(args.model_type)

        # Train and evaluate model
        print(f"\n🚀 Starting {args.model_type} training...")
        model, results = trainer.train_and_evaluate(X, y, test_size=0.2)

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
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
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
                'test_size': 0.2
            },
            'features': FEATURE_ORDER[:X.shape[1]],
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

if __name__ == '__main__':
    main()
