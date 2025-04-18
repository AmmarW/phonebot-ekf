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
        psi_pred = psi + wz * self.dt

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
        # (Further Jacobian terms can be added for psi & biases)
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
