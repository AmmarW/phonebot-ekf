import numpy as np

def trim_initial_static(accel_df, threshold=5, max_samples=7000):
    """
    Find end of initial static window by seeing when accel magnitude
    first deviates from its initial mean by more than `threshold`.
    If no deviation within max_samples, use max_samples as the window.
    
    accel_df: DataFrame indexed by seconds_elapsed with ['ax','ay','az']
    threshold: allowable g‐unit deviation (sensor units)
    max_samples: max rows to scan for a static window
    Returns: start_time (float seconds) for motion
    """
    # compute accel magnitude in sensor units
    a_vec = accel_df[['ax','ay','az']].to_numpy()
    a_mag = np.linalg.norm(a_vec, axis=1)
    
    # initial “static” level
    init = a_mag[0]
    # scan up to max_samples or end of data
    scan_len = min(len(a_mag), max_samples)
    for i in range(1, scan_len):
        if abs(a_mag[i] - init) > threshold:
            # motion starts here
            return accel_df.index[i]
    # if we never exceed threshold, start after our window
    return accel_df.index[scan_len-1]


def estimate_biases(static_accel_df, static_gyro_df):
    """
    Compute biases as the mean over a static window.
    If provided window is empty, raise an error.
    """
    if static_accel_df.empty or static_gyro_df.empty:
        raise ValueError("Static window is empty—cannot estimate biases.")
    # mean accel minus zero (we treat static accel as bias)
    accel_bias = static_accel_df[['ax','ay','az']].median().to_numpy()
    # mean gyro
    gyro_bias  = static_gyro_df[['gx','gy','gz']].median().to_numpy()
    return accel_bias, gyro_bias


def apply_calibration(accel_df, gyro_df, accel_bias, gyro_bias):
    """
    Subtract biases from the streams.
    """
    accel_df[['ax','ay','az']] -= accel_bias
    gyro_df [['gx','gy','gz']] -= gyro_bias
    return accel_df, gyro_df
