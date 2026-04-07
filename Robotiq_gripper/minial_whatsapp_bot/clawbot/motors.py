"""
Minimal motor abstractions for a clawbot simulation.

Provides a simple Motor class and a DifferentialDrive wrapper to set left/right RPMs.
"""

class Motor:
    def __init__(self, name: str, max_rpm: float = 100.0):
        self.name = name
        self.max_rpm = float(max_rpm)
        self.rpm = 0.0

    def set_rpm(self, rpm: float) -> float:
        # Clamp RPM to allowed range
        rpm = max(-self.max_rpm, min(self.max_rpm, rpm))
        self.rpm = float(rpm)
        return self.rpm

    def get_rpm(self) -> float:
        return self.rpm

class DifferentialDrive:
    def __init__(self, left: Motor, right: Motor):
        self.left = left
        self.right = right

    def set_wheel_rpm(self, left_rpm: float, right_rpm: float):
        self.left.set_rpm(left_rpm)
        self.right.set_rpm(right_rpm)
