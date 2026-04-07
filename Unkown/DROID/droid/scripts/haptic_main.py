from droid.controllers.haptic_touch_controller import HapticTouchPolicy
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI
import argparse

parser = argparse.ArgumentParser(description='Haptic device controller for robot control.')

# No left/right controller arguments needed for haptic device
args = parser.parse_args()

# Make the robot env (cameras will be skipped if not available)
env = RobotEnv()

# Create haptic touch controller
controller = HapticTouchPolicy()

# Make the data collector
data_collector = DataCollecter(env=env, controller=controller)

# Make the GUI
user_interface = RobotGUI(robot=data_collector, right_controller=False)




