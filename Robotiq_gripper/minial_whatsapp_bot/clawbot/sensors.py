"""
Simple encoder/sensor stubs for clawbot simulation.
"""

class Encoder:
    def __init__(self, ticks_per_rev: int = 360):
        self.ticks_per_rev = int(ticks_per_rev)
        self.position = 0.0  # in encoder ticks

    def tick(self, rpm: float, dt: float):
        """
        Simulate encoder progress given wheel RPM over a small time dt (seconds).
        rpm -> revolutions per minute. revolutions per second = rpm / 60.
        position increments by revolutions * ticks_per_rev.
        """
        if dt <= 0:
            return
        revolutions_per_sec = rpm / 60.0
        self.position += revolutions_per_sec * dt * self.ticks_per_rev

    def read(self) -> float:
        return float(self.position)
