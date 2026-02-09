import numpy as np
import cv2


class BallKalman:
    def __init__(self, fps=30):
        self.kf = cv2.KalmanFilter(4, 2)
        # 4 = Number of state variables (x, y, vx, vy)
        # 2 = Number of measurements (x, y)

        # time scaling factor
        self.dt = 1.0 / fps

        # Damping factor (friction/deceleration)
        # Prevents velocity from "exploding"
        damp = 0.999

        # State: [x, y, vx, vy]
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, self.dt, 0],  # x + vx * dt
                [0, 1, 0, self.dt],  # y + vy * dt
                [0, 0, damp, 0],  # vx
                [0, 0, 0, damp],  # vy
            ],
            np.float32,
        )

        # What YOLO Measures: (x, y)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            np.float32,
        )

        # Prediction noise covariance
        self.kf.processNoiseCov = np.diag(
            [1.0, 1.0, 2.0, 2.0]  # x noise  # y noise  # vx noise  # vy noise
        ).astype(np.float32)

        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, x, y, conf=None):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True

        if conf is not None:
            # Map confidence (0.0 to 1.0) to a noise value.
            # If conf is 1.0, noise is 0.05 (Very Trusting)
            # If conf is 0.1, noise is ~2.0 (Very Skeptical)
            noise_value = 0.05 / (conf + 1e-6)
            self.kf.measurementNoiseCov = (
                np.eye(2, dtype=np.float32) * noise_value
            ).astype(np.float32)
        else:
            # Fallback to default if no conf provided
            self.kf.measurementNoiseCov = (np.eye(2, dtype=np.float32) * 0.1).astype(
                np.float32
            )

        self.kf.predict()
        self.kf.correct(measurement)

        vx = np.clip(self.kf.statePost[2], -30, 30)
        vy = np.clip(self.kf.statePost[3], -30, 30)
        self.kf.statePost[2] = vx
        self.kf.statePost[3] = vy

    def predict(self):
        if not self.initialized:
            return 0, 0

        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])
