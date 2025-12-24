import pandas as pd
import numpy as np



def load_from_csv(filename, sep=';'):
    """
    Load data from CSV file.
    
    Args:
        filename (str): Path to CSV file
        sep (str): Column separator (default: ';')
    
    Returns:
        pd.DataFrame: Loaded data, or None if error
    """
    try:
        df = pd.read_csv(filename, sep=sep)
        print(f"Loaded {len(df)} rows from {filename}")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None



def clean_dataset(df, value_column, convert_percentage=False, drop_unnamed=True):
    """
    Clean dataset: handle percentages, remove NaN, set date index.
    
    Args:
        df: Raw dataframe
        value_column: Main data column
        convert_percentage: Convert "+2.5%" to 0.025
        drop_unnamed: Drop unnamed columns
    
    Returns:
        Cleaned dataframe with date index
    """
    df_clean = df.copy()
    
    # Drop unnamed columns if they exist
    if drop_unnamed:
        unnamed_cols = [col for col in df_clean.columns if 'Unnamed' in col]
        if unnamed_cols:
            df_clean = df_clean.drop(columns=unnamed_cols)
    
    # Convert percentage strings to decimals if needed
    if convert_percentage and df_clean[value_column].dtype == 'object':
        df_clean[value_column] = (
            df_clean[value_column]
            .str.replace('+', '', regex=False)
            .str.replace('%', '', regex=False)
            .astype(float) / 100
        )
    
    df_clean = df_clean.dropna(subset=[value_column])
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean = df_clean.set_index('date')
    df_clean = df_clean.dropna(how='all')
    
    return df_clean



def align_datasets_by_date(*dfs, how='inner'):
    """
    Align multiple datasets to common date range.
    
    Args:
        *dfs: Dataframes to align (must have date index)
        how: 'inner' (intersection) or 'outer' (union)
    
    Returns:
        Tuple of aligned dataframes
    """
    if how == 'inner':
        common_dates = dfs[0].index
        for df in dfs[1:]:
            common_dates = common_dates.intersection(df.index)
        aligned = tuple(df.loc[common_dates].sort_index() for df in dfs)
        
    elif how == 'outer':
        all_dates = dfs[0].index
        for df in dfs[1:]:
            all_dates = all_dates.union(df.index)
        aligned = tuple(df.reindex(all_dates).sort_index().fillna(method='ffill') for df in dfs)
    
    else:
        raise ValueError("how must be 'inner' or 'outer'")
    
    return aligned


def print_data_summary(df, name):
    """Print summary statistics for a cleaned dataset."""
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nLast 3 rows:")
    print(df.tail(3))


def load_all_data():
    """Load and clean all datasets."""
    
    df_spxtr = load_from_csv('01_data_raw/spxtr.csv')
    df_stoxx50gr = load_from_csv('01_data_raw/stoxx50gr_usd.csv')
    df_eurusd = load_from_csv('01_data_raw/eurusd.csv')
    df_fed = load_from_csv('01_data_raw/fed_rate.csv')
    df_ecb = load_from_csv('01_data_raw/ecb_rate.csv')
    df_vix = load_from_csv('01_data_raw/vix.csv')
    df_vstoxx = load_from_csv('01_data_raw/vstoxx.csv')
    df_us_tech = load_from_csv('01_data_raw/us_tech.csv')
    df_eu_tech = load_from_csv('01_data_raw/eu_tech.csv')
    
    df_spxtr = clean_dataset(df_spxtr, 'spxtr_return', convert_percentage=True)
    df_stoxx50gr = clean_dataset(df_stoxx50gr, 'stoxx50erusd_return', convert_percentage=True)
    df_eurusd = clean_dataset(df_eurusd, 'eurusd_midprice', drop_unnamed=False)
    df_fed = clean_dataset(df_fed, 'FEDFUNDS')
    df_ecb = clean_dataset(df_ecb, 'ecb_rate')
    df_vix = clean_dataset(df_vix, 'vixxmonthly')
    df_vstoxx = clean_dataset(df_vstoxx, 'v2txmonthly')
    df_us_tech = clean_dataset(df_us_tech, 'xlk_totalreturn', convert_percentage=False)
    df_eu_tech = clean_dataset(df_eu_tech, 'eu_tech_exv3', convert_percentage=False)
    
    df_us_tech['xlk_totalreturn'] = pd.to_numeric(df_us_tech['xlk_totalreturn'], errors='coerce')
    df_eu_tech['eu_tech_exv3'] = pd.to_numeric(df_eu_tech['eu_tech_exv3'], errors='coerce')
    
    df_stoxx50gr = df_stoxx50gr.rename(columns={'stoxx50erusd_return': 'stoxx50gr_return'})
    df_eurusd = df_eurusd.rename(columns={'eurusd_midprice': 'eurusd'})
    df_fed = df_fed.rename(columns={'FEDFUNDS': 'fed_rate'})
    df_vix = df_vix.rename(columns={'vixxmonthly': 'vix'})
    df_vstoxx = df_vstoxx.rename(columns={'v2txmonthly': 'vstoxx'})
    df_us_tech = df_us_tech.rename(columns={'xlk_totalreturn': 'us_tech_return'})
    df_eu_tech = df_eu_tech.rename(columns={'eu_tech_exv3': 'eu_tech_return'})
    
    df_all = pd.concat([df_spxtr, df_stoxx50gr, df_eurusd, df_fed, df_ecb, df_vix, df_vstoxx, df_us_tech, df_eu_tech], axis=1)
    
    return df_all