# filters.py
import numpy as np

class EMAFilter:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = np.array(x, dtype=float)
        else:
            self.value = self.alpha * np.array(x) + (1 - self.alpha) * self.value
        return tuple(self.value)

class SimpleKalman2D:
    def __init__(self, dt=1.0, process_var=1e-3, measure_var=1e-2):
        # state: [x, y, vx, vy]
        self.dt = dt
        self.x = np.zeros((4,1))
        self.P = np.eye(4) * 1.0
        # state transition
        self.F = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1,0],
                           [0,0,0,1]], dtype=float)
        # measurement: x,y
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * measure_var

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        # z: (x, y)
        z = np.array(z).reshape(2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x[:2].flatten()
