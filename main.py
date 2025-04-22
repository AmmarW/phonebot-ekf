import os
import pandas as pd
import numpy as np
from data_loader import load_csv, estimate_rate
from preprocessor import trim_initial_data, estimate_biases, apply_calibration
from utils import phone_to_robot, latlon_to_enu
from ekf_filter import EKF
import matplotlib.pyplot as plt

# --- Quaternion to Roll, Pitch, Yaw Conversion ---
def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion (w, x, y, z) to roll, pitch, yaw in radians."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx*qx + qy*qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * (np.pi / 2),
                     np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy*qy + qz*qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
# ------------------------------------------

def main():
    # Setup
    project_dir = 'phone_localization'
    os.makedirs(project_dir, exist_ok=True)

    # 1) Load sensor data
    accel = load_csv('tests/pavementFromthirdTree2secondTree/Accelerometer.csv')
    gyro  = load_csv('tests/pavementFromthirdTree2secondTree/Gyroscope.csv')
    gps   = load_csv('tests/pavementFromthirdTree2secondTree/Location.csv')
    orientation = load_csv('tests/pavementFromthirdTree2secondTree/Orientation.csv')

    # 2) Rename columns
    accel.rename(columns={'x':'ax', 'y':'ay', 'z':'az'}, inplace=True)
    gyro .rename(columns={'x':'gx', 'y':'gy', 'z':'gz'}, inplace=True)

    # 3) Print sampling rates
    print(f"Accel rate: {estimate_rate(accel):.2f} Hz")
    print(f"Gyro  rate: {estimate_rate(gyro):.2f} Hz")
    print(f"GPS   rate: {estimate_rate(gps):.2f} Hz")
    print(f"Orient rate: {estimate_rate(orientation):.2f} Hz")

    # 4) Trim initial unreliable data
    t0 = trim_initial_data(accel, gyro, gps)
    print(f"Trimming initial data to {t0}...")
    accel       = accel[accel.index >= t0]
    gyro        = gyro[gyro.index  >= t0]
    gps         = gps[gps.index   >= t0]
    print("First 3 rows of GPS data after trimming:")
    print(gps.head(3))
    orientation = orientation[orientation.index >= t0]

    # 5) Estimate and remove biases
    static_accel = accel[accel.index < t0]
    static_gyro  = gyro[gyro.index   < t0]
    try:
        accel_bias, gyro_bias = estimate_biases(static_accel, static_gyro)
    except ValueError:
        # Fallback: first N samples
        N = 200
        accel_bias = accel[['ax','ay','az']].iloc[:N].mean().to_numpy()
        gyro_bias  = gyro [['gx','gy','gz']].iloc[:N].mean().to_numpy()
    accel, gyro = apply_calibration(accel, gyro, accel_bias, gyro_bias)

    # 6) Transform to robot frame
    accel_arr = accel[['ax','ay','az']].values
    gyro_arr  = gyro [['gx','gy','gz']].values
    accel_rot, gyro_rot = phone_to_robot(accel_arr, gyro_arr)
    accel[['ax','ay','az']] = accel_rot
    gyro [['gx','gy','gz']] = gyro_rot

    # 7) Convert GPS lat/lon to local ENU (meters)
    lat0, lon0 = gps.iloc[0][['latitude','longitude']]
    enu = np.stack(
        gps.apply(lambda r: latlon_to_enu(r['latitude'], r['longitude'], lat0, lon0), axis=1).values
    )

    # 8) Initialize EKF
    dt    = 1.0 / estimate_rate(accel)
    Q     = np.eye(8) * 1e-5
    R_gps = np.eye(2) * (gps['horizontalAccuracy'].mean() ** 2) * 1.2
    print(R_gps)
    ekf   = EKF(dt, Q, R_gps)

    # 9) Run EKF predict/update
    state_hist = []
    raw_unix   = []
    gps_times  = gps.index.to_list()
    gps_pts    = enu
    gi = 0

    # Track GPS correction points
    gps_corrections = []

    for t, row in accel.iterrows():
        # IMU data
        ax, ay = row['ax'], row['ay']
        gz = gyro.at[t, 'gz']

        # Orientation from quaternion
        qw = orientation.at[t, 'qw']
        qx = orientation.at[t, 'qx']
        qy = orientation.at[t, 'qy']
        qz = orientation.at[t, 'qz']
        _, _, psi = quat_to_euler(qw, qx, qy, qz)

        # Predict
        ekf.predict(np.array([ax, ay]), gz, psi)

        # GPS correction
        if gi < len(gps_times) and abs((t - gps_times[gi]).total_seconds()) < dt / 2:
            ekf.update_gps(gps_pts[gi])
            gps_corrections.append((ekf.x.copy(), row['unix_time']))  # Save corrected state
            gi += 1

        state_hist.append(ekf.x.copy())
        raw_unix.append(row['unix_time'])

    # 10) Save and plot results
    states_df = pd.DataFrame(
        state_hist,
        index=raw_unix,
        columns=['px','py','vx','vy','psi','bax','bay','bwz']
    )
    states_df.index.name = 'unix_time'
    states_df.reset_index(inplace=True)

    out_path = os.path.join(project_dir, 'ekf_states.csv')
    states_df.to_csv(out_path, index=False)
    print(f"Done → EKF states written to {out_path}")


    # Extract GPS correction points for plotting
    gps_correction_points = pd.DataFrame(
        [state for state, _ in gps_corrections],
        columns=['px', 'py', 'vx', 'vy', 'psi', 'bax', 'bay', 'bwz']
    )

    # Plot trajectory with GPS corrections
    plt.figure(figsize=(8, 6))
    plt.plot(states_df['px'], states_df['py'], label='Trajectory', color='blue')
    # plt.scatter(
    #     gps_correction_points['px'], gps_correction_points['py'],
    #     color='red', label='GPS Corrections', zorder=5
    # )
    plt.xlabel('px (m)')
    plt.ylabel('py (m)')
    plt.title('Estimated 2D Trajectory with GPS Corrections')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    corrections = [
        (before, after) for before, (after, _) in zip(state_hist[:-1], gps_corrections)
    ]
    corrections_df = pd.DataFrame(
        corrections, columns=['before', 'after']
    )
    corrections_df['delta_px'] = corrections_df['after'].apply(lambda x: x[0]) - corrections_df['before'].apply(lambda x: x[0])
    corrections_df['delta_py'] = corrections_df['after'].apply(lambda x: x[1]) - corrections_df['before'].apply(lambda x: x[1])

    plt.figure(figsize=(8, 6))
    plt.plot(corrections_df['delta_px'], label='Delta px', color='red')
    plt.plot(corrections_df['delta_py'], label='Delta py')
    plt.title('Difference Between Predicted and Corrected States')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()