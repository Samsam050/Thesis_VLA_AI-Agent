"""
Demo script to exercise the clawbot simulation.
Prints a few iterations to stdout.
"""

from clawbot.motors import Motor, DifferentialDrive
from clawbot.sensors import Encoder
from clawbot.claw import Claw
from clawbot.control import PController


def run_demo(steps: int = 20, dt: float = 0.1):
    left = Motor("left", max_rpm=120)
    right = Motor("right", max_rpm=120)
    drive = DifferentialDrive(left, right)
    enc = Encoder(ticks_per_rev=360)
    claw = Claw()
    ctrl = PController(kp=2.0, target=100.0)

    t = 0.0
    for i in range(steps):
        speed = ctrl.update(enc.read(), dt)
        # apply the same speed to both wheels for straight motion
        drive.set_wheel_rpm(speed, speed)
        enc.tick(speed, dt)
        t += dt
        print(f"t={t:.2f}s, pos={enc.read():.1f} ticks, speed={speed:.1f} rpm")

if __name__ == "__main__":
    run_demo()
