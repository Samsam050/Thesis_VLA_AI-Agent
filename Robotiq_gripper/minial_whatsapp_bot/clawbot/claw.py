"""
Simple Claw actuator wrapper around a Motor.
"""

from .motors import Motor

class Claw:
    def __init__(self, motor: Motor = None):
        self.motor = motor or Motor("claw", max_rpm=100)
        self.state = "closed"

    def open(self, rpm: float = 50.0):
        # Move claw to open position by spinning in positive direction
        self.motor.set_rpm(abs(rpm))
        self.state = "open"

    def close(self, rpm: float = 50.0):
        self.motor.set_rpm(-abs(rpm))
        self.state = "closed"

    def stop(self):
        self.motor.set_rpm(0.0)
