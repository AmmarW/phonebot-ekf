# data_loader.py

import pandas as pd

def load_csv(file_path):
    """
    Loads a CSV that has a raw UNIX timestamp in the first column
    and sensor data in the others. Detects 'time' or 'timestamp',
    renames it to 'unix_time', parses it as datetime (ns vs s),
    sets it as the DataFrame index, and drops 'seconds_elapsed'.
    """
    df = pd.read_csv(file_path)

    # 1) Rename the raw time column
    if 'time' in df.columns:
        df.rename(columns={'time': 'unix_time'}, inplace=True)
    elif 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'unix_time'}, inplace=True)
    else:
        raise KeyError(f"No 'time' or 'timestamp' column found in {file_path}")

    # 2) Parse UNIX → datetime
    ut0 = df['unix_time'].iloc[0]
    # If it's a huge int (ns), else assume seconds
    unit = 'ns' if ut0 > 1e12 else 's'
    df['datetime'] = pd.to_datetime(df['unix_time'], unit=unit)

    # 3) Index by datetime
    df.set_index('datetime', inplace=True)

    # 4) Drop the obsolete seconds_elapsed column if present
    if 'seconds_elapsed' in df.columns:
        df.drop(columns=['seconds_elapsed'], inplace=True)

    return df


def estimate_rate(df):
    """
    Compute sampling rate (Hz) from a datetime index
    or numeric index.
    """
    idx = df.index
    if pd.api.types.is_datetime64_any_dtype(idx):
        dt = idx.to_series().diff().dt.total_seconds().dropna()
    else:
        dt = idx.to_series().diff().dropna()
    median_dt = dt.median()
    return 1.0 / median_dt if median_dt > 0 else float('nan')
