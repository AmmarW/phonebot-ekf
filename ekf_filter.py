import numpy as np

class EKF:
    def __init__(self, dt, Q, R_gps):
        self.dt = dt
        self.x = np.zeros(8)  # [px, py, vx, vy, psi, bax, bay, bwz]
        self.P = np.eye(8) * 1e-3
        self.Q = Q
        self.R_gps = R_gps

    def predict(self, accel_xy, gyro_z):
        # Unpack biases
        bax, bay, bwz = self.x[5], self.x[6], self.x[7]
        ax, ay = accel_xy - np.array([bax, bay])
        wz = gyro_z - bwz

        # Unpack state
        px, py, vx, vy, psi = self.x[:5]

        # Predict orientation
        psi_pred = psi + self.dt*wz

        # Rotation matrix
        c, s = np.cos(psi), np.sin(psi)
        R = np.array([[c, -s], [s, c]])

        # Predict velocity & position
        a_world = R.dot(np.array([ax, ay]))
        vx_pred = vx + a_world[0] * self.dt
        vy_pred = vy + a_world[1] * self.dt
        px_pred = px + vx_pred * self.dt
        py_pred = py + vy_pred * self.dt

        # State prediction
        x_pred = np.array([px_pred, py_pred, vx_pred, vy_pred, psi_pred, bax, bay, bwz])

        # Covariance prediction
        F = np.eye(8)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        # F[2, 4] = -ax*s*self.dt - ay*c*self.dt
        # F[3, 4] = ax*c*self.dt - ay*s*self.dt
        self.P = F.dot(self.P).dot(F.T) + self.Q
        self.x = x_pred

    def update_gps(self, z):
        H = np.zeros((2, 8))
        H[0, 0] = 1
        H[1, 1] = 1
        y = z - self.x[:2]
        S = H.dot(self.P).dot(H.T) + self.R_gps
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(8) - K.dot(H)).dot(self.P)
# import numpy as np

# class EKF:
#     def __init__(self, dt, Q, R_gps):
#         self.dt = dt
#         self.x = np.zeros(8)  # [px, py, vx, vy, psi, bax, bay, bwz]
#         self.P = np.eye(8) * 1e-3
#         self.Q = Q
#         self.R_gps = R_gps

#     def normalize_angle(self, angle):
#         """Wrap angle to [-pi, pi]"""
#         return (angle + np.pi) % (2 * np.pi) - np.pi

#     def predict(self, accel_xy, gyro_z):
#         # Unpack current state and biases
#         px, py, vx, vy, psi, bax, bay, bwz = self.x
#         ax, ay = accel_xy - np.array([bax, bay])
#         wz = gyro_z - bwz

#         # Predict new orientation
#         psi_pred = psi + wz * self.dt
#         psi_pred = self.normalize_angle(psi_pred)

#         # Rotation matrix with updated heading
#         c, s = np.cos(psi_pred), np.sin(psi_pred)
#         R = np.array([[c, -s],
#                       [s,  c]])

#         # Acceleration in world frame
#         a_world = R.dot(np.array([ax, ay]))

#         # Predict new velocity
#         vx_pred = vx + a_world[0] * self.dt
#         vy_pred = vy + a_world[1] * self.dt

#         # Predict new position
#         px_pred = px + vx_pred * self.dt
#         py_pred = py + vy_pred * self.dt

#         # State prediction
#         self.x = np.array([px_pred, py_pred, vx_pred, vy_pred, psi_pred, bax, bay, bwz])

#         # Jacobian matrix F (partial update)
#         F = np.eye(8)
#         F[0, 2] = self.dt  # ∂px/∂vx
#         F[1, 3] = self.dt  # ∂py/∂vy

#         # ∂vx/∂psi and ∂vy/∂psi (due to rotation of accel)
#         dR_dpsi = np.array([[-s, -c],
#                             [ c, -s]])  # ∂R/∂psi
#         da_dpsi = dR_dpsi.dot(np.array([ax, ay]))
#         F[2, 4] = da_dpsi[0] * self.dt
#         F[3, 4] = da_dpsi[1] * self.dt

#         # ∂vx/∂bax and ∂vy/∂bay (biases affect acceleration)
#         F[2, 5] = -R[0, 0] * self.dt
#         F[2, 6] = -R[0, 1] * self.dt
#         F[3, 5] = -R[1, 0] * self.dt
#         F[3, 6] = -R[1, 1] * self.dt

#         # ∂psi/∂bwz
#         F[4, 7] = -self.dt

#         # Covariance prediction
#         self.P = F @ self.P @ F.T + self.Q

#     def update_gps(self, z):
#         if np.any(np.isnan(z)):
#             return  # Skip update if GPS reading is invalid

#         # Measurement model H
#         H = np.zeros((2, 8))
#         H[0, 0] = 1  # px
#         H[1, 1] = 1  # py

#         # Innovation
#         y = z - self.x[:2]

#         # Kalman gain
#         S = H @ self.P @ H.T + self.R_gps
#         K = self.P @ H.T @ np.linalg.inv(S)

#         # Update state
#         self.x += K @ y
#         self.x[4] = self.normalize_angle(self.x[4])  # Normalize psi

#         # Update covariance
#         I = np.eye(8)
#         self.P = (I - K @ H) @ self.P
