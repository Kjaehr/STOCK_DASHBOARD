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
from sklearn.preprocessing import StandardScaler
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
        self.feature_importance = {}
        
    def create_models(self) -> Dict[str, Any]:
        """Create individual models for ensemble"""
        models = {
            'logistic': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
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
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        if HAS_XGBOOST and model_name == 'xgboost':
            param_grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
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
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and tune individual models
        base_models = self.create_models()
        tuned_models = []
        
        for name, model in base_models.items():
            print(f"Tuning {name}...")
            tuned_model = self.hyperparameter_tuning(X_scaled, y, name, model)
            tuned_models.append((name, tuned_model))
            self.models[name] = tuned_model
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=tuned_models,
            voting='soft'  # Use probabilities
        )
        
        print("Training ensemble...")
        ensemble.fit(X_scaled, y)
        
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
    """Compute basic technical features from OHLCV data"""
    df = df.copy()

    # SMAs
    df['sma20'] = df['Close'].rolling(window=20).mean()
    df['sma50'] = df['Close'].rolling(window=50).mean()
    df['sma200'] = df['Close'].rolling(window=200).mean()

    # Price ratios
    df['price_over_sma20'] = df['Close'] / df['sma20']
    df['price_over_sma50'] = df['Close'] / df['sma50']
    df['price_over_sma200'] = df['Close'] / df['sma200']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_norm'] = df['rsi'] / 100.0

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['Close']

    # ATR buckets
    atr_quantiles = df['atr_pct'].quantile([0.2, 0.4, 0.6, 0.8])
    df['atr_bucket_0'] = (df['atr_pct'] <= atr_quantiles.iloc[0]).astype(int)
    df['atr_bucket_1'] = ((df['atr_pct'] > atr_quantiles.iloc[0]) &
                         (df['atr_pct'] <= atr_quantiles.iloc[1])).astype(int)
    df['atr_bucket_2'] = ((df['atr_pct'] > atr_quantiles.iloc[1]) &
                         (df['atr_pct'] <= atr_quantiles.iloc[2])).astype(int)
    df['atr_bucket_3'] = ((df['atr_pct'] > atr_quantiles.iloc[2]) &
                         (df['atr_pct'] <= atr_quantiles.iloc[3])).astype(int)
    df['atr_bucket_4'] = (df['atr_pct'] > atr_quantiles.iloc[3]).astype(int)

    # Additional features
    df['vol20_rising'] = (df['Volume'].rolling(20).mean() >
                         df['Volume'].rolling(20).mean().shift(5)).astype(int)
    df['price_gt_ma20'] = (df['Close'] > df['sma20']).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_momentum'] = df['rsi'].diff(5)
    df['sma_alignment'] = ((df['sma20'] > df['sma50']) &
                          (df['sma50'] > df['sma200'])).astype(int)
    df['above_all_smas'] = ((df['Close'] > df['sma20']) &
                           (df['Close'] > df['sma50']) &
                           (df['Close'] > df['sma200'])).astype(int)
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['price_momentum_5d'] = df['Close'].pct_change(5)
    df['price_momentum_10d'] = df['Close'].pct_change(10)

    # Bollinger Bands position
    bb_middle = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

    # Volume trend
    df['volume_trend'] = df['Volume'].rolling(10).mean() / df['Volume'].rolling(30).mean()

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

        # For now, create a simple model metadata (actual training to be implemented)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
        model_file = out_dir / f"model_{args.version}_{timestamp}_ensemble.json"

        # Calculate basic stats
        if args.label_type == 'multiclass':
            unique_labels, counts = np.unique(y, return_counts=True)
            label_dist = dict(zip(unique_labels, counts.tolist()))
        else:
            label_dist = {"mean": float(np.mean(y)), "std": float(np.std(y))}

        # Save model metadata
        model_metadata = {
            'version': args.version,
            'model_type': args.model_type,
            'label_type': args.label_type,
            'parameters': {
                'years': args.years,
                'horizon': args.horizon,
                'max_tickers': args.max_tickers
            },
            'features': FEATURE_ORDER[:22],  # Current features only
            'timestamp': timestamp,
            'dataset_info': {
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'n_tickers': len(tickers),
                'label_distribution': label_dist
            },
            'performance': {
                'accuracy': 0.0,  # Placeholder - implement actual training
                'auc': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        }

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

        print(f"✅ Model metadata saved to {model_file}")
        print(f"Dataset: {X.shape[0]} samples from {len(tickers)} tickers")
        print("🚀 Ready for actual model training implementation!")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == '__main__':
    main()
