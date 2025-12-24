import pandas as pd
import numpy as np


def create_return_differential(df, us_return_col, eu_return_col):
    """Create return differential (Y variable)."""
    return df[us_return_col] - df[eu_return_col]


def create_fx_change(df, fx_col):
    """Create FX rate change variable (monthly percentage change)."""
    return df[fx_col].pct_change(fill_method=None)


def create_policy_differential(df, us_rate_col, eu_rate_col):
    """Create monetary policy differential (US - EU)."""
    return df[us_rate_col] - df[eu_rate_col]


def create_volatility_differential(df, us_vol_col, eu_vol_col):
    """Create volatility differential (US - EU)."""
    return df[us_vol_col] - df[eu_vol_col]


def create_tech_differential(df, us_tech_col, eu_tech_col):
    """Create tech sector performance differential (US - EU)."""
    return df[us_tech_col] - df[eu_tech_col]


def create_all_features(df):
    """
    Create all features for the model.
    
    Args:
        df: DataFrame with all input variables
    
    Returns:
        DataFrame with Y, X1, X2, X3, X4 features
    """
    df_features = df.copy()
    
    df_features['Y'] = create_return_differential(df, 'spxtr_return', 'stoxx50gr_return')
    
    df_features['X1'] = create_fx_change(df, 'eurusd')
    
    df_features['X2'] = create_policy_differential(df, 'fed_rate', 'ecb_rate')
    
    df_features['X3'] = create_volatility_differential(df, 'vix', 'vstoxx')
    
    df_features['X4'] = create_tech_differential(df, 'us_tech_return', 'eu_tech_return')
    
    df_features = df_features[['Y', 'X1', 'X2', 'X3', 'X4']].dropna()
    
    return df_features
