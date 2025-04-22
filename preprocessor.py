import numpy as np
import pandas as pd


def trim_initial_data(accel_df, gyro_df, gps_df,
                      gps_accuracy_thresh=5.0,
                      accel_var_thresh=0.05,
                      gyro_var_thresh=0.01,
                      window_seconds=2.0):
    """
    Trim the start of data until:
      1) GPS horizontalAccuracy ≤ gps_accuracy_thresh
      2) Rolling std of accel magnitude ≤ accel_var_thresh
      3) Rolling std of gyro magnitude  ≤ gyro_var_thresh
    Each computed over a sliding window of window_seconds.

    Returns the timestamp where all conditions are first met.
    """
    # 1) Find first "good" GPS timestamp
    if 'horizontalAccuracy' in gps_df.columns:
        good_gps = gps_df[gps_df['horizontalAccuracy'] <= gps_accuracy_thresh]
        t_gps = good_gps.index[0] if not good_gps.empty else gps_df.index[0]
    else:
        t_gps = gps_df.index[0]

    # 2) Compute window size in samples based on accel sampling
    dt = accel_df.index.to_series().diff().dt.total_seconds().dropna()
    median_dt = dt.median() if not dt.empty else window_seconds
    window = max(int(window_seconds / median_dt), 1)

    # 3) Compute rolling std of accel and gyro magnitudes
    a_mag = np.linalg.norm(accel_df[['ax', 'ay', 'az']].values, axis=1)
    a_std = pd.Series(a_mag, index=accel_df.index).rolling(window, min_periods=1).std()
    print(f"Accel rolling std: {a_std}")

    g_mag = np.linalg.norm(gyro_df[['gx', 'gy', 'gz']].values, axis=1)
    g_std = pd.Series(g_mag, index=gyro_df.index).rolling(window, min_periods=1).std()

    print(f"Gyro rolling std: {g_std.reindex_like(a_std)}")

    # 4) Find first time both accel and gyro are stable
    stable_mask = (a_std <= accel_var_thresh) & (g_std.reindex_like(a_std) <= gyro_var_thresh)
    stable_times = stable_mask[stable_mask].index
    t_stable = stable_times[0] if not stable_times.empty else accel_df.index[0]

    # 5) Return the later of GPS-good and sensor-stable times
    return max(t_gps, t_stable)


def estimate_biases(static_accel_df, static_gyro_df):
    """
    Compute biases as the median over a static window.
    """
    if static_accel_df.empty or static_gyro_df.empty:
        raise ValueError("Static window is empty—cannot estimate biases.")
    accel_bias = static_accel_df[['ax', 'ay', 'az']].median().to_numpy()
    gyro_bias  = static_gyro_df[['gx', 'gy', 'gz']].median().to_numpy()
    return accel_bias, gyro_bias


def apply_calibration(accel_df, gyro_df, accel_bias, gyro_bias):
    """
    Subtract biases from accelerometer and gyroscope dataframes.
    """
    accel_df[['ax', 'ay', 'az']] -= accel_bias
    gyro_df [['gx', 'gy', 'gz']] -= gyro_bias
    return accel_df, gyro_df
