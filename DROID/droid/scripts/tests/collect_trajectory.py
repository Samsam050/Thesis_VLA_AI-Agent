from droid.controllers.haptic_touch_controller import HapticTouchPolicy
from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import collect_trajectory

# Make the robot env
env = RobotEnv()
controller = HapticTouchPolicy()

print("env and haptic device setup correctly!")
collect_trajectory(env, controller=controller)
