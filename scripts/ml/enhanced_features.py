#!/usr/bin/env python3
"""
Enhanced feature engineering for better ML models.
Adds technical indicators, fundamental ratios, and sentiment features.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import yfinance as yf

def add_advanced_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced technical indicators"""
    df = df.copy()
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Williams %R
    df['williams_r'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
    
    # Commodity Channel Index (CCI)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma_tp) / (0.015 * mad)
    
    # Average Directional Index (ADX) - simplified
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr_14'] = true_range.rolling(window=14).mean()
    
    # Price momentum
    df['momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
    
    # VWAP
    df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['price_vs_vwap'] = df['Close'] / df['vwap'] - 1
    
    # Volume indicators
    df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_sma_ratio'] = df['Volume'] / df['volume_sma_20']
    
    # Volatility rank (percentile of volatility over 252 days)
    df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()
    df['volatility_rank'] = df['volatility_20d'].rolling(window=252).rank(pct=True)
    
    return df

def add_fundamental_features(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add fundamental analysis features"""
    df = df.copy()
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Financial ratios (normalized to 0-1 range)
        pe_ratio = info.get('trailingPE', np.nan)
        pb_ratio = info.get('priceToBook', np.nan)
        debt_to_equity = info.get('debtToEquity', np.nan)
        roe = info.get('returnOnEquity', np.nan)
        
        # Normalize ratios (simple min-max scaling with reasonable bounds)
        df['pe_ratio_norm'] = np.clip(pe_ratio / 50.0, 0, 1) if pe_ratio else 0.5
        df['pb_ratio_norm'] = np.clip(pb_ratio / 10.0, 0, 1) if pb_ratio else 0.5
        df['debt_to_equity_norm'] = np.clip(debt_to_equity / 200.0, 0, 1) if debt_to_equity else 0.5
        df['roe_norm'] = np.clip(roe / 0.3, 0, 1) if roe else 0.5
        
        # Growth metrics
        revenue_growth = info.get('revenueGrowth', np.nan)
        earnings_growth = info.get('earningsGrowth', np.nan)
        
        df['revenue_growth_norm'] = np.clip((revenue_growth + 0.5) / 1.0, 0, 1) if revenue_growth else 0.5
        df['earnings_growth_norm'] = np.clip((earnings_growth + 0.5) / 1.0, 0, 1) if earnings_growth else 0.5
        
    except Exception as e:
        print(f"Warning: Could not fetch fundamentals for {ticker}: {e}")
        # Fill with neutral values
        for col in ['pe_ratio_norm', 'pb_ratio_norm', 'debt_to_equity_norm', 
                   'roe_norm', 'revenue_growth_norm', 'earnings_growth_norm']:
            df[col] = 0.5
    
    return df

def add_sentiment_features(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment analysis features (placeholder for now)"""
    df = df.copy()
    
    # Placeholder sentiment features (to be implemented with real news data)
    # These would come from news sentiment analysis
    df['news_sentiment_1d'] = 0.5  # Neutral sentiment
    df['news_sentiment_7d'] = 0.5
    df['news_sentiment_30d'] = 0.5
    df['social_sentiment'] = 0.5
    df['analyst_sentiment'] = 0.5
    
    return df

def add_sector_features(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add sector and industry momentum features"""
    df = df.copy()
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Placeholder for sector momentum (would need sector ETF data)
        df['sector_momentum'] = 0.5
        df['industry_momentum'] = 0.5
        df['relative_strength_sector'] = 0.5
        
    except Exception as e:
        print(f"Warning: Could not fetch sector info for {ticker}: {e}")
        df['sector_momentum'] = 0.5
        df['industry_momentum'] = 0.5
        df['relative_strength_sector'] = 0.5
    
    return df

def create_multi_class_labels(df: pd.DataFrame, horizon: int = 20, 
                            thresholds: Dict[str, float] = None) -> pd.Series:
    """Create multi-class labels for better prediction granularity"""
    if thresholds is None:
        thresholds = {
            'strong_up': 0.10,    # >10%
            'weak_up': 0.02,      # 2-10%
            'sideways': 0.02,     # -2% to 2%
            'weak_down': 0.10,    # -10% to -2%
            'strong_down': 0.10   # <-10%
        }
    
    # Calculate future returns
    future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
    
    # Create labels
    labels = pd.Series(index=df.index, dtype='category')
    
    labels[future_returns >= thresholds['strong_up']] = 'STRONG_UP'
    labels[(future_returns >= thresholds['weak_up']) & 
           (future_returns < thresholds['strong_up'])] = 'WEAK_UP'
    labels[(future_returns >= -thresholds['sideways']) & 
           (future_returns < thresholds['weak_up'])] = 'SIDEWAYS'
    labels[(future_returns >= -thresholds['weak_down']) & 
           (future_returns < -thresholds['sideways'])] = 'WEAK_DOWN'
    labels[future_returns < -thresholds['weak_down']] = 'STRONG_DOWN'
    
    return labels

def create_regression_targets(df: pd.DataFrame, horizons: list = [5, 20, 60]) -> pd.DataFrame:
    """Create regression targets for multiple time horizons"""
    targets = pd.DataFrame(index=df.index)
    
    for horizon in horizons:
        # Simple future return
        targets[f'return_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1
        
        # Risk-adjusted return (return / volatility)
        volatility = df['Close'].pct_change().rolling(window=20).std()
        targets[f'risk_adj_return_{horizon}d'] = targets[f'return_{horizon}d'] / volatility
        
        # Maximum favorable excursion (best return within horizon)
        future_prices = pd.concat([df['Close'].shift(-i) for i in range(1, horizon+1)], axis=1)
        max_return = future_prices.max(axis=1) / df['Close'] - 1
        targets[f'max_return_{horizon}d'] = max_return
    
    return targets

def engineer_all_features(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps"""
    print(f"Engineering features for {ticker}...")
    
    # Add all feature types
    df = add_advanced_technical_features(df)
    df = add_fundamental_features(ticker, df)
    df = add_sentiment_features(ticker, df)
    df = add_sector_features(ticker, df)
    
    # Fill any remaining NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    return df

if __name__ == '__main__':
    # Test with a sample ticker
    ticker = 'AAPL'
    stock = yf.Ticker(ticker)
    df = stock.history(period='2y')
    
    # Engineer features
    enhanced_df = engineer_all_features(ticker, df)
    
    # Create labels
    multi_labels = create_multi_class_labels(enhanced_df)
    regression_targets = create_regression_targets(enhanced_df)
    
    print(f"Enhanced features shape: {enhanced_df.shape}")
    print(f"Multi-class label distribution:")
    print(multi_labels.value_counts())
    print(f"Regression targets shape: {regression_targets.shape}")
