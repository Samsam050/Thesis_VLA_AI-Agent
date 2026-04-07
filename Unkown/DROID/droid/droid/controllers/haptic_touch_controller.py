import time
import numpy as np
from threading import Lock

from droid.controllers.pyOpenHaptics import hd
from droid.controllers.pyOpenHaptics.hd_callback import hd_callback
from droid.controllers.pyOpenHaptics.hd_device import HapticDevice

from droid.misc.subprocess_utils import run_threaded_command
from droid.misc.transformations import add_angles, euler_to_quat, quat_diff, quat_to_euler, rmat_to_quat

try:
    import open3d as o3d
except ImportError:
    o3d = None


import sys
import termios
import tty
import select
KEYBOARD_AVAILABLE = True
import os
import signal


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


def create_coordinate_frame(pos, quat, size=0.1):
    from scipy.spatial.transform import Rotation as R
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    frame.transform(T)
    return frame


class HapticDeviceState:
    def __init__(self):
        self.position = np.zeros(3)
        self.transform = np.eye(4)
        self.button = 0 # 1: the lower pressed; 2: the higher pressed; 3: both pressed; 0: both free
        # 1: enable movement; 2: reset orientation; 3: movement + close gripper
        self.lock = Lock()

    def update(self, position, transform, button):
        with self.lock:
            self.position = np.array(position)
            self.transform = transform.copy()
            self.button = button

    def get_state(self):
        with self.lock:
            return self.position.copy(), self.transform.copy(), self.button


g_haptic_state = HapticDeviceState()


@hd_callback
def haptic_callback():
    transform = hd.get_transform()
    position = [transform[3][0], transform[3][1], transform[3][2]]
    matrix = np.array(transform).reshape((4, 4)).T # column major to row major
    # Disable rotation by setting rotation part to identity
    # matrix[:3, :3] = np.eye(3)
    button = hd.get_buttons()
    g_haptic_state.update(position, matrix, button)
    # print("The pose is ", position)


class HapticTouchPolicy:
    def __init__(
        self,
        max_lin_vel=1.0,
        max_rot_vel=1.0,
        max_gripper_vel=1.0,
        spatial_coeff=0.001,
        pos_action_gain=10.0,
        rot_action_gain=2.0,
        gripper_action_gain=3.0,
        # rmat_reorder=[3, 1, 2, 4],
        rmat_reorder=[-3, -1, 2, 4],
        enable_visualization=False,
        robot_interface=None,
    ):
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.haptic_to_global_mat = np.eye(4)
        self.robot_interface = robot_interface

        self.device = None
        self.device_ok = False
        self._shutdown = False  # Flag to signal threads to stop
        self.reset_orientation = True
        self.reset_state()

        self.vis_lock = Lock()
        self.vis_data = {"robot_pos": None, "robot_quat": None, "haptic_pos": None, "haptic_quat": None}
        self.vis_enabled = enable_visualization and o3d is not None
        if self.vis_enabled:
            run_threaded_command(self._visualization_loop)

        try:
            print("[HapticTouchPolicy] Attempting to initialize haptic device...")
            self.device = HapticDevice(callback=haptic_callback, scheduler_type="async")
            # hd.start_scheduler()
            # hdAsyncSheduler(haptic_callback)
            self.device_ok = True
            print("[HapticTouchPolicy] Haptic device initialized successfully")
        except Exception as e:
            print(f"[HapticTouchPolicy] Haptic device init failed: {e}")

        print("[HapticTouchPolicy] Starting state update thread...")
        run_threaded_command(self._update_state)

        if KEYBOARD_AVAILABLE:
            print("[HapticTouchPolicy] Starting keyboard listener thread...")
            run_threaded_command(self._keyboard_listener)
        else:
            print("[HapticTouchPolicy] Keyboard listener not available (termios not found)")

    def reset_state(self):
        self._state = {
            "button": 0,
            "movement_enabled": False,
            "controller_on": True,
            "success": False, 
            "keyboard_gripper":0.0,
        }
        self.update_sensor = True
        self.reset_origin = True
        self.robot_origin = None
        self.haptic_origin = None
        self._process_reading()
    
    def _send_immediate_gripper_command(self):
        """
        Forces an update of the internal state so forward() picks it up instantly.
        """
        self._process_reading()

    def _keyboard_listener(self):
        """Listen for keyboard input to set success state."""
        print("[Keyboard] isatty:", sys.stdin.isatty())

        if not KEYBOARD_AVAILABLE:
            return

        print("[HapticTouchPolicy] Keyboard listener started. Press 's' to mark success, 'r' to reset success.")

        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            # Set terminal to raw mode for non-blocking input
            tty.setraw(fd)

            while not self._shutdown:
                # Non-blocking read with timeout
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)

                    if char == '\x03' or char.lower() == 'q':
                        print("\r\n[Haptic] Ctrl+C detected. Exiting...")
                        self._shutdown = True
                        # Trigger system interrupt to kill main process if needed
                        os.kill(os.getpid(), signal.SIGINT)
                        break

                    if char.lower() == 's':
                        self._state["success"] = True
                        print("\r\n[HapticTouchPolicy] Success (via keyboard 's')")
                    elif char.lower() == 'r':
                        self._state["failure"] = True
                        print("\r\n[HapticTouchPolicy] RESET (via keyboard 'r')")
                    elif char.lower() == 'c':
                        self._state["keyboard_gripper"] = 1.0
                        print("\r\n[HapticTouchPolicy] Gripper CLOSE (keyboard 'c')")
                        self._send_immediate_gripper_command()

                    elif char.lower() == 'o':
                        self._state["keyboard_gripper"] = 0.0
                        print("\r\n[HapticTouchPolicy] Gripper OPEN (keyboard 'o')")
                        self._send_immediate_gripper_command()
                time.sleep(0.05)
        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _update_state(self, num_wait_sec=5, hz=50):
        last_read = time.time()
        print(f"[HapticTouchPolicy] State update loop started (hz={hz})")
        while not self._shutdown:
            time.sleep(1.0 / hz)

            if not self.device_ok:
                continue

            _, transform, button = g_haptic_state.get_state()

            self._state["controller_on"] = (time.time() - last_read) < num_wait_sec

            button_1_only = button == 1
            button_2_only = button == 2
            #both_buttons = button == 3

            #movement_enabled = button_1_only or both_buttons
            movement_enabled = button_1_only
            toggled = self._state["movement_enabled"] != movement_enabled
            self.update_sensor = self.update_sensor or movement_enabled
            self.reset_orientation = self.reset_orientation or button_2_only
            self.reset_origin = self.reset_origin or toggled

            if toggled:
                print(f"[HapticTouchPolicy] Movement toggled - enabled: {movement_enabled}")

            if button != self._state["button"]:
                print(f"[HapticTouchPolicy] Button state changed: {self._state['button']} -> {button}")

            self._state["button"] = button
            self._state["movement_enabled"] = movement_enabled
            last_read = time.time()

            if self.reset_orientation:
                if button_2_only or self._state["movement_enabled"]:
                    self.reset_orientation = False
                    print("[HapticTouchPolicy] Orientation reset completed")
                try:
                    self.haptic_to_global_mat = np.linalg.inv(transform)
                except:
                    self.haptic_to_global_mat = np.eye(4)
                    self.reset_orientation = True
                    print("[HapticTouchPolicy] Failed to invert transform matrix, using identity")

    def _process_reading(self):
        _, transform, button = g_haptic_state.get_state()
        rot_mat = self.global_to_env_mat @ self.haptic_to_global_mat @ transform
        pos = self.spatial_coeff * rot_mat[:3, 3]
        quat = rmat_to_quat(rot_mat[:3, :3])
        #gripper = 1.0 if button == 3 else 0.0
        gripper = self._state["keyboard_gripper"]
        if gripper is None:
            gripper = 0.0
        
        self.haptic_state = {"pos": pos, "quat": quat, "gripper": gripper}
        # print(f"[HapticTouchPolicy] Processed reading - pos: {pos}, gripper: {gripper}")

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        lin_norm = np.linalg.norm(lin_vel)
        rot_norm = np.linalg.norm(rot_vel)
        if lin_norm > self.max_lin_vel:
            print("The pos v is too large")
            lin_vel *= self.max_lin_vel / lin_norm
        if rot_norm > self.max_rot_vel:
            rot_vel *= self.max_rot_vel / rot_norm
        if abs(gripper_vel) > self.max_gripper_vel:
            gripper_vel = np.sign(gripper_vel) * self.max_gripper_vel
        return lin_vel, rot_vel, gripper_vel

    def _visualization_loop(self, hz=30):
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Haptic & Robot", width=800, height=600)

            # World frame (white)
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            vis.add_geometry(world_frame)

            # Robot base frame (gray, static)
            robot_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            vis.add_geometry(robot_base)

            # Robot end-effector and haptic frames (will be updated)
            robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            haptic_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(robot_frame)
            vis.add_geometry(haptic_frame)

            print("[HapticTouchPolicy] Visualization window opened")

            while not self._shutdown:
                update = False
                with self.vis_lock:
                    if self.vis_data["robot_pos"] is not None:
                        robot_pos = self.vis_data["robot_pos"]
                        robot_quat = self.vis_data["robot_quat"]
                        haptic_pos = self.vis_data["haptic_pos"]
                        haptic_quat = self.vis_data["haptic_quat"]
                        update = True

                if update:
                    vis.remove_geometry(robot_frame, False)
                    vis.remove_geometry(haptic_frame, False)
                    robot_frame = create_coordinate_frame(robot_pos, robot_quat, 0.1)
                    haptic_frame = create_coordinate_frame(haptic_pos, haptic_quat, 0.1)
                    vis.add_geometry(robot_frame, False)
                    vis.add_geometry(haptic_frame, False)

                vis.poll_events()
                vis.update_renderer()
                time.sleep(1.0 / hz)

            vis.destroy_window()
            print("[HapticTouchPolicy] Visualization window closed")
        except Exception as e:
            print(f"[HapticTouchPolicy] Visualization error: {e}")

    def _update_visualization(self, robot_pos, robot_quat, haptic_pos, haptic_quat):
        if not self.vis_enabled:
            return
        with self.vis_lock:
            self.vis_data["robot_pos"] = robot_pos.copy()
            self.vis_data["robot_quat"] = robot_quat.copy()
            self.vis_data["haptic_pos"] = haptic_pos.copy()
            self.vis_data["haptic_quat"] = haptic_quat.copy()

    def _calculate_action(self, state_dict, include_info=False):
        """
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False
        """
        self._process_reading()
        
        robot_pos = np.array(state_dict["cartesian_position"][:3])
        robot_euler = state_dict["cartesian_position"][3:]
        robot_quat = euler_to_quat(robot_euler)
        robot_gripper = state_dict["gripper_position"]

        if self.reset_origin:
            self.robot_origin = {"pos": robot_pos, "quat": robot_quat}
            self.haptic_origin = {"pos": self.haptic_state["pos"], "quat": self.haptic_state["quat"]}
            self.reset_origin = False
            # print(f"[HapticTouchPolicy] Origin reset - robot_pos: {robot_pos}, haptic_pos: {self.haptic_state['pos']}")

        # Only calculate translation and orientation actions when movement is enabled (button 1 pressed)
        if self._state["movement_enabled"]:
            robot_pos_offset = robot_pos - self.robot_origin["pos"]
            target_pos_offset = self.haptic_state["pos"] - self.haptic_origin["pos"]
            pos_action = target_pos_offset - robot_pos_offset

            robot_quat_offset = quat_diff(robot_quat, self.robot_origin["quat"])
            target_quat_offset = quat_diff(self.haptic_state["quat"], self.haptic_origin["quat"])
            quat_action = quat_diff(target_quat_offset, robot_quat_offset)
            euler_action = quat_to_euler(quat_action)
        else:
            pos_action = np.zeros(3)
            euler_action = np.zeros(3)

        # print("The detected gripper is", self.haptic_state["gripper"], "the current robot gripper is", robot_gripper)
        gripper_action = (self.haptic_state["gripper"] * 1.5) - robot_gripper
        

        target_pos = pos_action + robot_pos
        target_euler = add_angles(euler_action, robot_euler)
        target_cartesian = np.concatenate([target_pos, target_euler])

        pos_action *= self.pos_action_gain
        euler_action *= self.rot_action_gain
        gripper_action *= self.gripper_action_gain
        lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action, gripper_action)

        action = np.concatenate([lin_vel, rot_vel, [gripper_vel]]).clip(-1, 1)

        self._update_visualization(robot_pos, robot_quat, self.haptic_state["pos"], self.haptic_state["quat"])

        if include_info:
            info = {
                "target_cartesian_position": target_cartesian,
                "target_gripper_position": self.haptic_state["gripper"]
            }
            return action, info
        return action

    def get_info(self):
        return {
            "success": self._state["success"],  # Set by keyboard 's' or both buttons
            "failure": False,
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }

    def forward(self, obs_dict, include_info=False):
        if not self.device_ok or self.haptic_state is None:
            if not self.device_ok:
                print("[HapticTouchPolicy] Device not OK, returning zero action")
            if self.haptic_state is None:
                print("[HapticTouchPolicy] Haptic state is None, returning zero action")
            return (np.zeros(7), {}) if include_info else np.zeros(7)
        return self._calculate_action(obs_dict["robot_state"], include_info=include_info)

    def close(self):
        """Properly shutdown the haptic device and threads."""
        print("[HapticTouchPolicy] Shutting down...")
        self._shutdown = True
        time.sleep(0.2)

        if self.device:
            try:
                print("[HapticTouchPolicy] Closing haptic device...")
                self.device.close()
                self.device = None
                self.device_ok = False
                print("[HapticTouchPolicy] Device closed successfully")
            except Exception as e:
                print(f"[HapticTouchPolicy] Error during device close: {e}")

    def __del__(self):
        """Destructor - ensure cleanup happens even if close() wasn't called."""
        if not self._shutdown:
            self.close()
