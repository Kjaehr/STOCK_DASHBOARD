#!/usr/bin/env python3
"""
Enhanced ML training with XGBoost, Random Forest, and ensemble methods.
Supports multi-class labels, multiple timeframes, and advanced validation.

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
import yfinance as yf
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
    
    # TODO: Implement data loading and feature engineering
    # This is a template - actual implementation would load data from your existing pipeline
    
    # Placeholder for now
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
    model_file = out_dir / f"model_{args.version}_{timestamp}_ensemble.json"
    
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
        'performance': {
            'accuracy': 0.0,  # Placeholder
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
    
    print(f"Model template saved to {model_file}")
    print("Next steps:")
    print("1. Integrate with existing data pipeline")
    print("2. Implement feature engineering")
    print("3. Add model training logic")
    print("4. Implement validation and backtesting")

if __name__ == '__main__':
    main()
