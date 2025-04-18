# main.py

import os
import pandas as pd
import numpy as np
from data_loader import load_csv, estimate_rate
from preprocessor import trim_initial_static, estimate_biases, apply_calibration
from utils import phone_to_robot, latlon_to_enu
from ekf_filter import EKF

def main():
    project_dir = 'phone_localization'
    os.makedirs(project_dir, exist_ok=True)

    # 1) Load CSVs; index is now datetime parsed from raw UNIX ns/s
    accel = load_csv('tests/pavementFromthirdTree2secondTree/Accelerometer.csv')
    gyro  = load_csv('tests/pavementFromthirdTree2secondTree/Gyroscope.csv')
    gps   = load_csv('tests/pavementFromthirdTree2secondTree/Location.csv')

    # 2) Rename axes columns for EKF
    accel.rename(columns={'x':'ax','y':'ay','z':'az'}, inplace=True)
    gyro .rename(columns={'x':'gx','y':'gy','z':'gz'}, inplace=True)

    # 3) Estimate rates
    print(f"Accel rate: {estimate_rate(accel):.2f} Hz")
    print(f"Gyro  rate: {estimate_rate(gyro):.2f} Hz")
    print(f"GPS   rate: {estimate_rate(gps):.2f} Hz")

    # 4) Trim initial static period
    t0 = trim_initial_static(accel)
    orig_accel = accel.copy()
    orig_gyro  = gyro.copy()
    accel = accel[accel.index >= t0]
    gyro  = gyro [gyro.index  >= t0]
    gps   = gps  [gps.index   >= t0]
    print(f"Trimmed start @ {t0} → accel {accel.shape}, gyro {gyro.shape}, gps {gps.shape}")

    # 5) Estimate & remove biases (with fallback if needed)
    static_accel = orig_accel[orig_accel.index < t0]
    static_gyro  = orig_gyro [orig_gyro .index < t0]
    try:
        accel_bias, gyro_bias = estimate_biases(static_accel, static_gyro)
    except ValueError:
        # fallback to first 200 samples
        N = 200
        accel_bias = orig_accel[['ax','ay','az']].iloc[:N].mean().to_numpy()
        gyro_bias  = orig_gyro [['gx','gy','gz']].iloc[:N].mean().to_numpy()
        print(f"⚠️  Static window empty—using first {N} samples for bias")

    apply_calibration(accel, gyro, accel_bias, gyro_bias)
    print("Biases removed, sample accel:\n", accel[['ax','ay','az']].head())

    # 6) Rotate phone→robot frame
    accel_arr = accel[['ax','ay','az']].values
    gyro_arr  = gyro [['gx','gy','gz']].values
    accel_rot, gyro_rot = phone_to_robot(accel_arr, gyro_arr)
    accel[['ax','ay','az']] = accel_rot
    gyro [['gx','gy','gz']] = gyro_rot

    # 7) Build local ENU from GPS lat/lon
    lat0, lon0 = gps.iloc[0][['latitude','longitude']]
    enu = np.stack(gps.apply(
        lambda r: latlon_to_enu(r['latitude'], r['longitude'], lat0, lon0),
        axis=1
    ).values)

    # 8) Initialize EKF
    dt    = 1.0 / estimate_rate(accel)
    Q     = np.eye(8) * 1e-3
    R_gps = np.eye(2) * (gps['horizontalAccuracy'].mean()**2)
    ekf   = EKF(dt, Q, R_gps)

    # 9) Run predict/update, recording raw UNIX alongside states
    state_hist = []
    raw_unix   = []
    gps_times  = gps.index.to_list()
    gps_pts    = enu
    gi = 0

    for ax, ay, gz, t in zip(accel['ax'], accel['ay'], gyro['gz'], accel.index):
        ekf.predict(np.array([ax, ay]), gz)

        # GPS update if timestamp aligns within dt/2
        if gi < len(gps_times) and abs((t - gps_times[gi]).total_seconds()) < dt/2:
            ekf.update_gps(gps_pts[gi])
            gi += 1

        state_hist.append(ekf.x.copy())
        # grab the raw UNIX timestamp from the DataFrame
        raw_unix.append(accel['unix_time'].loc[t])

    # 10) Build and save output with raw UNIX first column
    states_df = pd.DataFrame(
        state_hist,
        index=raw_unix,
        columns=['px','py','vx','vy','psi','bax','bay','bwz']
    )
    states_df.index.name = 'unix_time'
    states_df.reset_index(inplace=True)

    # write CSV
    out_path = os.path.join(project_dir, 'ekf_states.csv')
    states_df.to_csv(out_path, index=False)
    print(f"Done → fused states written to {out_path}")
    
if __name__ == '__main__':
    main()
