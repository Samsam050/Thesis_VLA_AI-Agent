"""
A tiny PID-like controller for demonstration purposes.
"""

class PController:
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, target: float = 0.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.target = float(target)

        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, measurement: float, dt: float) -> float:
        """
        Compute control output given a measurement and delta time.
        """
        if dt <= 0:
            return 0.0
        error = self.target - measurement
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative
