# Robot Localization
> *This project was developed as part of MEEN 700: Robotic Perception*

This project implements **on-board localization for a mobile robot** using data collected from a **smartphone mounted on the robot platform**. The localization system leverages an **Extended Kalman Filter (EKF)** to estimate the robot's 2D position and orientation in real-time using fused **IMU (accelerometer and gyroscope)** and **GPS** data. This approach provides a lightweight, quick-to-deploy alternative to traditional ROS-based localization pipelines.

---

## Features

- Sensor fusion using EKF with an 8-state vector: `[px, py, vx, vy, ψ, bax, bay, bwz]`.
- Input from smartphone sensors via the [Sensor Logger app](https://www.tszheichoi.com/sensorlogger).
- Conversion of IMU data from phone frame to robot body frame using rotation matrices.
- Automatic trimming of stationary periods and bias estimation from static IMU readings.
- Visualization and benchmarking using ground-truth comparisons (if available).

---

## System Architecture

The system includes:
- **Python-based EKF pipeline** for prediction and correction.
- **Preprocessing** for sensor calibration and frame alignment.
- **Visualization** for plotting estimated trajectories.

### Folder Structure

```
├── main.py              # Entry point for executing the pipeline
├── data_loader.py       # CSV loading and timestamp alignment
├── preprocessor.py      # Static bias estimation, trimming, calibration
├── utils.py             # Frame transformation and GPS to ENU conversion
├── ekf_filter.py        # Core EKF logic (prediction + GPS update)
├── tests/               # Sample datasets with IMU + GPS
├── phone_localization/  # Output CSVs (results)
```

---

## Usage

### 1. Prepare Data  
Place your sensor logs (CSV format) in the `tests/` directory. Files must include:
- Accelerometer and gyroscope (100 Hz)
- Orientation (yaw)
- GPS (lat, lon, horizontal accuracy, 1 Hz)

### 2. Run Localization  
Run the following command to execute the pipeline:
```bash
python main.py
```

### 3. Review Results  
Output files will be generated in the `phone_localization/` directory:

- **ekf_states.csv**:  
    - `px, py`: Estimated positions  
    - `vx, vy`: Estimated velocities  
    - `ψ`: Orientation (yaw)  
    - `bax, bay, bwz`: Estimated biases  

- **Plots**: Visualized trajectories for validation.

---

## Results & Benchmarks

- Estimated paths closely match actual movement across tested configurations.
- Robust to GPS dropout (1 Hz) using high-frequency IMU integration.
- Sensor alignment and pre-processing significantly improve accuracy.

---

## Known Issues

- **GPS Cold Start**: TTFF (time to first fix) may exceed one hour on new modules.
- **Magnetic Interference**: Phone compass can be unreliable near motors/LiDAR.
- **Mount Stability**: Wind and vibration can affect sensor readings.
- **Frame Alignment**: Misaligned frames reduce accuracy if not corrected properly.

---

## Future Improvements

- Adaptive tuning of EKF noise models for varying environments.
- Integration with ROS for real-time robotic deployments.
- Extended support for visual odometry or wheel encoders.

---

## Contributors

- **Abdulaziz Alharbi** – Hardware setup, data acquisition, and documentation  
- **Ammar Waheed** – System design, EKF implementation, and documentation  
- **Hassan Niaz** – Mathematical modeling and EKF development  

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

