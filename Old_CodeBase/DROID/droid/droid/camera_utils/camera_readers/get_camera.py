from copy import deepcopy
import threading
import time

import cv2
import numpy as np
import depthai as dai
import pyrealsense2 as rs

from droid.misc.parameters import hand_camera_id, varied_camera_1_id
from droid.misc.time import time_ms


resize_func_map = {"cv2": cv2.resize, None: None}

# Output sizes for the adapter layer. These are output frame sizes, not claims about
# the underlying sensor being a real stereo RGB camera.

standard_params = dict(camera_resolution=(640, 480), camera_fps=52)
#standard_params = dict(camera_resolution=(640, 350), camera_fps=30)
advanced_params = dict(camera_resolution=(2054, 1520), camera_fps=30)


def gather_cameras():
    all_cameras = []
    target_serials = [hand_camera_id, varied_camera_1_id]

    for serial in target_serials:
        if not serial:
            continue
        if serial == hand_camera_id:
            try:
                print("Identified as OAK-D (Hand). Connecting...")
                cam = OakCamera(serial)
                all_cameras.append(cam)
                print("OAK-D Connected!")
            except Exception as e:
                print(f"FAILED to open OAK-D {serial}: {e}")
        else:
            try:
                print("Identified as RealSense (Body). Connecting...")
                cam = RealSenseCamera(serial)
                all_cameras.append(cam)
                print("RealSense Connected!")
            except Exception as e:
                print(f"FAILED to open RealSense {serial}: {e}")

    return all_cameras


class OakCamera:
    """
    OAK-D hand-camera adapter for a ZED-shaped interface.

    Important: this uses CAM_A (real color camera) only. The returned *_left and
    *_right images are the same RGB/BGR frame duplicated so downstream code that
    expects ZED-style keys can continue working.
    """

    def __init__(self, serial_number):
        self.serial_number = str(serial_number)
        self.is_hand_camera = True
        self.high_res_calibration = False
        self.current_mode = None
        self._current_params = None
        self._intrinsics = {}

        self.pipeline = None
        self.device = None
        self.q_rgb = None
        self.thread = None

        self.frame = None
        self.running = False
        self.latency = 0

        self.traj_image = True
        self.traj_concatenate_images = False
        self.traj_resolution = (0, 0)
        self.depth = False
        self.pointcloud = False
        self.resize_func = None

        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.resizer_resolution = (0, 0)

        print(f"Opening OAK-D (CAM_A adapter): {self.serial_number}")

    def enable_advanced_calibration(self):
        self.high_res_calibration = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution
        self.depth = depth
        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    def set_calibration_mode(self):
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.resizer_resolution = (0, 0)

        change_settings_1 = self.high_res_calibration and (self._current_params != advanced_params)
        change_settings_2 = (not self.high_res_calibration) and (self._current_params != standard_params)

        if change_settings_1:
            self._configure_camera(advanced_params)
        if change_settings_2:
            self._configure_camera(standard_params)

        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])
        self.resizer_resolution = self.traj_resolution

        if self._current_params != standard_params:
            self._configure_camera(standard_params)

        self.current_mode = "trajectory"

    def _open_depthai_device(self):
        """
        Try opening the requested OAK by serial number first. If that constructor
        path is not supported in the installed DepthAI build, fall back to the
        default device.
        """
        try:
            return dai.Device(dai.DeviceInfo(self.serial_number))
        except Exception:
            return dai.Device()

    def _configure_camera(self, init_params):
        self.disable_camera()

        try:
            self.device = self._open_depthai_device()
            self.pipeline = dai.Pipeline(self.device)

            w, h = init_params["camera_resolution"]
            fps = init_params["camera_fps"]

            cam_rgb = self.pipeline.create(dai.node.Camera)
            cam_rgb.build(dai.CameraBoardSocket.CAM_A)

            # Use interleaved BGR because getCvFrame() is easiest with OpenCV in this format.
            self.q_rgb = cam_rgb.requestOutput(
                (w, h),
                type=dai.ImgFrame.Type.BGR888p,
                enableUndistortion = True,
                fps=fps,
            ).createOutputQueue(maxSize=1, blocking=False)

            self._current_params = init_params
            self.latency = int(2.5 * (1e3 / fps))
            self.pipeline.start()

            calib_data = self.device.readCalibration()
            K = np.array(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h), dtype=np.float32)
            D = np.array(calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), dtype=np.float32)

            # Duplicate CAM_A intrinsics into left/right keys so downstream ZED-shaped
            # code can keep working.
            self._intrinsics = {
                self.serial_number + "_left": {"cameraMatrix": K.copy(), "distCoeffs": D[:5].copy()},
                self.serial_number + "_right": {"cameraMatrix": K.copy(), "distCoeffs": D[:5].copy()},
            }

            self.frame = np.zeros((h, w, 3), dtype=np.uint8)
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

        except Exception as e:
            self.disable_camera()
            raise RuntimeError(f"OAK-D Camera Failed To Open: {e}")

    def _update(self):
        while self.running:
            try:
                packet = self.q_rgb.tryGet() if self.q_rgb is not None else None
                if packet is not None:
                    self.frame = packet.getCvFrame().copy()
                time.sleep(0.001)
            except Exception:
                time.sleep(0.01)

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    def start_recording(self, filename):
        print("Warning: recording not implemented for OAK-D adapter")

    def stop_recording(self):
        pass

    def _process_frame(self, frame):
        frame = frame.copy()
        if self.resizer_resolution == (0, 0):
            return frame
        if self.resize_func is None:
            return cv2.resize(frame, self.resizer_resolution)
        return self.resize_func(frame, self.resizer_resolution)

    def read_camera(self):
        if self.skip_reading:
            return {}, {}

        if self.frame is None:
            return None

        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        now = time_ms()
        timestamp_dict[self.serial_number + "_read_end"] = now
        timestamp_dict[self.serial_number + "_frame_received"] = now
        timestamp_dict[self.serial_number + "_estimated_capture"] = now - self.latency

        data_dict = {}
        if self.image:
            frame = self.frame.copy()
            if self.concatenate_images:
                sbs_img = np.concatenate((frame, frame), axis=1)
                data_dict["image"] = {self.serial_number: self._process_frame(sbs_img)}
            else:
                processed = self._process_frame(frame)
                data_dict["image"] = {
                    self.serial_number + "_left": processed.copy(),
                    self.serial_number + "_right": processed.copy(),
                }

        return data_dict, timestamp_dict

    def disable_camera(self):
        if self.current_mode == "disabled" and self.device is None and self.pipeline is None:
            return

        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.thread = None

        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass

        self.device = None
        self.pipeline = None
        self.q_rgb = None
        self.frame = None
        self._current_params = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"


class RealSenseCamera:
    def __init__(self, serial_number):
        self.serial_number = str(serial_number)
        self.is_hand_camera = False
        self.high_res_calibration = False
        self.current_mode = None
        self._intrinsics = {}
        self.latency = 0

        self.frame = None
        self.pipeline = None
        self.profile = None
        self.thread = None
        self.running = True

        self.traj_image = True
        self.traj_concatenate_images = False
        self.traj_resolution = (0, 0)
        self.depth = False
        self.pointcloud = False
        self.resize_func = None

        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.resizer_resolution = (0, 0)

        print(f"Opening RealSense: {self.serial_number}")

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)
            #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.profile = self.pipeline.start(config)

            try:
                color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = color_stream.get_intrinsics()
                K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
                D = np.array(intr.coeffs[:5], dtype=np.float32)
                self._intrinsics = {
                    self.serial_number + "_left": {"cameraMatrix": K.copy(), "distCoeffs": D.copy()},
                    self.serial_number + "_right": {"cameraMatrix": K.copy(), "distCoeffs": D.copy()},
                }
            except Exception:
                K = np.eye(3, dtype=np.float32)
                D = np.zeros(5, dtype=np.float32)
                self._intrinsics = {
                    self.serial_number + "_left": {"cameraMatrix": K.copy(), "distCoeffs": D.copy()},
                    self.serial_number + "_right": {"cameraMatrix": K.copy(), "distCoeffs": D.copy()},
                }

            dev = self.profile.get_device()
            sensor = dev.first_color_sensor()
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            if sensor.supports(rs.option.backlight_compensation):
                sensor.set_option(rs.option.backlight_compensation, 1)

            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            self.current_mode = "trajectory"

        except Exception as e:
            self.disable_camera()
            raise RuntimeError(f"RealSense Failed To Open: {e}")

    def enable_advanced_calibration(self):
        self.high_res_calibration = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution
        self.depth = depth
        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    def set_calibration_mode(self):
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.resizer_resolution = (0, 0)
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])
        self.resizer_resolution = self.traj_resolution
        self.current_mode = "trajectory"

    def _update(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=500)
                color = frames.get_color_frame()
                if color:
                    self.frame = np.asanyarray(color.get_data()).copy()
            except RuntimeError:
                continue
            except Exception:
                time.sleep(0.01)

    def _process_frame(self, frame):
        frame = frame.copy()
        if self.resizer_resolution == (0, 0):
            return frame
        if self.resize_func is None:
            return cv2.resize(frame, self.resizer_resolution)
        return self.resize_func(frame, self.resizer_resolution)

    def read_camera(self):
        if self.skip_reading:
            return {}, {}

        if self.frame is None:
            return None

        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        now = time_ms()
        timestamp_dict[self.serial_number + "_read_end"] = now
        timestamp_dict[self.serial_number + "_frame_received"] = now
        timestamp_dict[self.serial_number + "_estimated_capture"] = now - self.latency

        frame = self.frame.copy()
        if self.concatenate_images:
            sbs_img = np.concatenate((frame, frame), axis=1)
            data = {"image": {self.serial_number: self._process_frame(sbs_img)}}
        else:
            processed = self._process_frame(frame)
            data = {
                "image": {
                    self.serial_number + "_left": processed.copy(),
                    self.serial_number + "_right": processed.copy(),
                }
            }

        return data, timestamp_dict

    def disable_camera(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.thread = None

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.pipeline = None
        self.profile = None
        self.frame = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    def start_recording(self, filename):
        pass

    def stop_recording(self):
        pass