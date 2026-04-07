from copy import deepcopy
import configparser
import glob
import os
import re
import subprocess
import urllib.request

import cv2
import numpy as np

from droid.misc.parameters import hand_camera_id
from droid.misc.time import time_ms


def gather_zed_cameras():
    all_zed_cameras = []
    seen_devices = set()

    for device_path in _list_video_devices():
        props = _get_udev_properties(device_path)

        should_try = _looks_like_zed_device(device_path, props)
        if not should_try:
            should_try = _probe_zed_like_device(device_path)
        if not should_try:
            continue

        physical_key = _get_physical_camera_key(device_path)
        if physical_key in seen_devices:
            continue

        serial = _get_serial_from_device(device_path)
        if serial is None:
            serial = _find_existing_calibration_serial()
        if serial is None:
            print(f"Skipping {device_path}: could not determine numeric ZED serial")
            continue

        cam = ZedCamera(device_path=device_path, serial_override=serial)
        all_zed_cameras.append(cam)
        seen_devices.add(physical_key)
        print(f"Found ZED camera at {device_path}, serial={cam.serial_number}")

    return all_zed_cameras


resize_func_map = {"cv2": cv2.resize, None: None}

# Side-by-side stereo resolutions:
# HD720  -> 2560 x 720  total  (1280 x 720 per eye)
# HD2K   -> 4416 x 1242 total  (2208 x 1242 per eye)
standard_params = dict(
    camera_resolution=(2560, 720),
    camera_fps=30,
)

advanced_params = dict(
    camera_resolution=(4416, 1242),
    camera_fps=15,
)


class ZedCamera:
    def __init__(self, device_path, serial_override=None):
        # Save Parameters #
        self.device_path = device_path
        self.device_index = _device_index_from_path(device_path)
        self.serial_number = str(serial_override) if serial_override is not None else str(self.device_index)
        self.is_hand_camera = self.serial_number == hand_camera_id
        self.high_res_calibration = False
        self.current_mode = None
        self._current_params = None
        self._extriniscs = {}
        self.software_contrast = 1.04
        self.software_brightness = 4
        self.software_saturation = 1.08
        self.software_gamma = 0.90
        self.software_sharpness = 0.06

        # OpenCV / calibration state #
        self._cap = None
        self._map_left_x = None
        self._map_left_y = None
        self._map_right_x = None
        self._map_right_y = None
        self._raw_intrinsics = {}
        self._projection_matrices = {}
        self._capture_eye_resolution = (0, 0)
        self._calibration_file = None

        # Reading parameter defaults #
        self.traj_image = True
        self.traj_concatenate_images = False
        self.traj_resolution = (0, 0)
        self.depth = False
        self.pointcloud = False
        self.resize_func = None

        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.zed_resolution = (0, 0)
        self.resizer_resolution = (0, 0)
        self.latency = 0

        # Open Camera #
        print("Opening Zed: ", self.serial_number)

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
        # Non-Permenant Values #
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution

        # Permenant Values #
        self.depth = depth
        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    ### Camera Modes ###
    def set_calibration_mode(self):
        # Set Parameters #
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.zed_resolution = (0, 0)
        self.resizer_resolution = (0, 0)

        # Set Mode #
        change_settings_1 = (self.high_res_calibration) and (self._current_params != advanced_params)
        change_settings_2 = (not self.high_res_calibration) and (self._current_params != standard_params)
        if change_settings_1:
            self._configure_camera(advanced_params)
        if change_settings_2:
            self._configure_camera(standard_params)
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        # Set Parameters #
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])

        # In the no-SDK backend we always capture at the active camera mode
        # and resize afterwards when a trajectory resolution is requested.
        self.zed_resolution = (0, 0)
        self.resizer_resolution = self.traj_resolution

        # Set Mode #
        change_settings = self._current_params != standard_params
        if change_settings:
            self._configure_camera(standard_params)
        self.current_mode = "trajectory"

    def _configure_camera(self, init_params):
        # Close Existing Camera #
        self.disable_camera()

        # Open Camera #
        self._current_params = deepcopy(init_params)
        requested_total_w, requested_h = init_params["camera_resolution"]
        requested_fps = init_params["camera_fps"]

        self._cap = self._open_capture(self.device_path, requested_total_w, requested_h, requested_fps)
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera Failed To Open")

        actual_total_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or float(requested_fps)

        if actual_total_w <= 0 or actual_h <= 0 or actual_total_w % 2 != 0:
            self.disable_camera()
            raise RuntimeError(f"Unexpected camera frame size: {actual_total_w}x{actual_h}")

        actual_eye_w = actual_total_w // 2
        self._capture_eye_resolution = (actual_eye_w, actual_h)

        print(
            f"ZED resolution: requested={requested_total_w}x{requested_h}@{requested_fps}, "
            f"actual={actual_total_w}x{actual_h}@{actual_fps:.2f}"
        )

        # Save Intrinsics #
        self.latency = int(2.5 * (1e3 / max(actual_fps, 1.0)))
        self._setup_calibration(actual_eye_w, actual_h)
        self.current_mode = "trajectory"

        # Flush a few startup frames so we do not begin with stale buffered data.
        for _ in range(3):
            self._cap.grab()

    def _setup_calibration(self, eye_w, eye_h):
        calibration_file = _download_calibration_file(self.serial_number)
        if calibration_file == "":
            raise RuntimeError(f"Could not load calibration file for ZED serial {self.serial_number}")

        (
            raw_left_matrix,
            raw_right_matrix,
            raw_left_dist,
            raw_right_dist,
            rect_left_matrix,
            rect_right_matrix,
            map_left_x,
            map_left_y,
            map_right_x,
            map_right_y,
            P1,
            P2,
        ) = _init_calibration(calibration_file, eye_w, eye_h)

        self._calibration_file = calibration_file
        self._map_left_x = map_left_x
        self._map_left_y = map_left_y
        self._map_right_x = map_right_x
        self._map_right_y = map_right_y

        # Store both raw and rectified calibration. The SDK LEFT/RIGHT views are
        # rectified, so get_intrinsics() returns rectified intrinsics with zero distortion.
        self._raw_intrinsics = {
            self.serial_number + "_left": self._process_intrinsics(raw_left_matrix, raw_left_dist),
            self.serial_number + "_right": self._process_intrinsics(raw_right_matrix, raw_right_dist),
        }

        self._projection_matrices = {
            self.serial_number + "_left": P1.copy(),
            self.serial_number + "_right": P2.copy(),
        }

        self._intrinsics = {
            self.serial_number + "_left": self._process_intrinsics(rect_left_matrix, np.zeros(5, dtype=np.float64)),
            self.serial_number + "_right": self._process_intrinsics(rect_right_matrix, np.zeros(5, dtype=np.float64)),
        }

    ### Calibration Utilities ###
    def _process_intrinsics(self, camera_matrix, dist_coeffs):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array(camera_matrix, dtype=np.float64).copy()
        intrinsics["distCoeffs"] = np.array(dist_coeffs, dtype=np.float64).reshape(-1).copy()
        return intrinsics

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    ### Recording Utilities ###
    def start_recording(self, filename):
        print("Warning: SVO recording is not available in the no-SDK backend. Skipping start_recording().")

    def stop_recording(self):
        pass

    ### Basic Camera Utilities ###
    def _process_frame(self, frame):
        frame = frame.copy()
        if self.resizer_resolution == (0, 0):
            return frame

        if self.resize_func is not None:
            return self.resize_func(frame, self.resizer_resolution)

        return cv2.resize(frame, self.resizer_resolution)

    def read_camera(self):
        # Skip if Read Unnecesary #
        if self.skip_reading:
            return {}, {}

        if self._cap is None or not self._cap.isOpened():
            return None

        # Read Camera #
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        timestamp_dict[self.serial_number + "_read_end"] = time_ms()

        # Benchmark Latency #
        received_time = timestamp_dict[self.serial_number + "_read_end"]
        timestamp_dict[self.serial_number + "_frame_received"] = received_time
        timestamp_dict[self.serial_number + "_estimated_capture"] = received_time - self.latency

        # Return Data #
        data_dict = {}

        if self.image:
            total_h, total_w = frame.shape[:2]
            if total_w % 2 != 0:
                return None

            left_raw = frame[:, : total_w // 2]
            right_raw = frame[:, total_w // 2 :]

            # Rebuild calibration maps if the runtime frame size differs from what we calibrated for.
            if (left_raw.shape[1], left_raw.shape[0]) != self._capture_eye_resolution:
                self._capture_eye_resolution = (left_raw.shape[1], left_raw.shape[0])
                self._setup_calibration(left_raw.shape[1], left_raw.shape[0])

            left_rect = cv2.remap(left_raw, self._map_left_x, self._map_left_y, interpolation=cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, self._map_right_x, self._map_right_y, interpolation=cv2.INTER_LINEAR)

            if self.concatenate_images:
                sbs_rect = np.concatenate([left_rect, right_rect], axis=1)
                data_dict["image"] = {self.serial_number: self._process_frame(sbs_rect)}
            else:
                data_dict["image"] = {
                    self.serial_number + "_left": self._process_frame(left_rect),
                    self.serial_number + "_right": self._process_frame(right_rect),
                }
        # if self.depth:
        # 	data_dict['depth'] = {}
        # if self.pointcloud:
        # 	data_dict['pointcloud'] = {}

        return data_dict, timestamp_dict

    def disable_camera(self):
        if self.current_mode == "disabled":
            return
        if self._cap is not None:
            self._current_params = None
            self._cap.release()
            self._cap = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"

    def _open_capture(self, device_path, width, height, fps):
        backends = [cv2.CAP_V4L2, None]
        fourccs = ["MJPG", "YUYV", None]
        best_cap = None

        for backend in backends:
            for fourcc in fourccs:
                cap = self._make_capture(device_path, backend)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue

                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                if fourcc is not None:
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                    except Exception:
                        pass

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)

                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if actual_w > 0 and actual_h > 0:
                    if actual_w == width and actual_h == height:
                        print(f"Using {device_path} with backend={backend} fourcc={fourcc}")
                        return cap

                    if best_cap is None:
                        best_cap = cap
                    else:
                        cap.release()
                else:
                    cap.release()

        if best_cap is not None:
            print(f"Warning: {device_path} did not honor the requested mode exactly; using best available mode instead.")
            return best_cap

        return None

    def _make_capture(self, device_path, backend):
        try:
            if backend is None:
                return cv2.VideoCapture(device_path)
            return cv2.VideoCapture(device_path, backend)
        except Exception:
            return None


def _list_video_devices():
    def sort_key(path):
        try:
            return int(path.replace("/dev/video", ""))
        except ValueError:
            return 10**9

    return sorted(glob.glob("/dev/video*"), key=sort_key)


def _device_index_from_path(device_path):
    return int(device_path.replace("/dev/video", ""))


def _get_udev_properties(device_path):
    props = {}
    try:
        output = subprocess.check_output(
            ["udevadm", "info", "--query=property", "--name", device_path],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return props

    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        props[key.strip()] = value.strip()

    return props


def _normalize_numeric_serial(value):
    if value is None:
        return None

    value = str(value).strip()
    if value.isdigit():
        return value

    match = re.search(r"\b(\d{6,})\b", value)
    if match:
        return match.group(1)

    return None


def _get_serial_from_props(props):
    candidates = [
        props.get("ID_SERIAL_SHORT"),
        props.get("ID_USB_SERIAL_SHORT"),
        props.get("ID_SERIAL"),
    ]

    for candidate in candidates:
        serial = _normalize_numeric_serial(candidate)
        if serial is not None:
            return serial

    return None


def _find_usb_root_for_video_device(device_path):
    try:
        sysfs_path = os.path.realpath(f"/sys/class/video4linux/video{_device_index_from_path(device_path)}/device")
    except Exception:
        return None

    current = sysfs_path
    visited = set()
    while current and current not in visited:
        visited.add(current)

        vendor_file = os.path.join(current, "idVendor")
        product_file = os.path.join(current, "idProduct")
        if os.path.isfile(vendor_file) and os.path.isfile(product_file):
            return current

        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return None


def _find_existing_calibration_serial():
    candidate_dirs = [
        os.path.join(os.path.expanduser("~"), "zed", "settings"),
        "/usr/local/zed/settings",
    ]

    matches = []
    for directory in candidate_dirs:
        for path in glob.glob(os.path.join(directory, "SN*.conf")):
            name = os.path.basename(path)
            match = re.match(r"SN(\d+)\.conf$", name)
            if match:
                matches.append(match.group(1))

    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]

    return None


def _get_serial_from_device(device_path):
    usb_root = _find_usb_root_for_video_device(device_path)
    if usb_root is not None:
        serial_file = os.path.join(usb_root, "serial")
        if os.path.isfile(serial_file):
            try:
                with open(serial_file, "r", encoding="utf-8") as f:
                    serial = _normalize_numeric_serial(f.read().strip())
                if serial is not None:
                    return serial
            except Exception:
                pass

    props = _get_udev_properties(device_path)
    serial = _get_serial_from_props(props)
    if serial is not None:
        return serial

    serial = _find_existing_calibration_serial()
    if serial is not None:
        return serial

    return None


def _get_physical_camera_key(device_path):
    usb_root = _find_usb_root_for_video_device(device_path)
    if usb_root is not None:
        return usb_root
    return device_path


def _looks_like_zed_device(device_path, props):
    haystack = " ".join([device_path] + list(props.values())).lower()

    if "stereolabs" in haystack:
        return True
    if "zed" in haystack and "video" in device_path:
        return True

    # Fallback for cases where udev text is sparse.
    try:
        video_name_path = f"/sys/class/video4linux/video{_device_index_from_path(device_path)}/name"
        with open(video_name_path, "r", encoding="utf-8") as f:
            name = f.read().strip().lower()
        if "stereolabs" in name or "zed" in name:
            return True
    except Exception:
        pass

    return False


def _probe_zed_like_device(device_path):
    test_modes = [
        (2560, 720, 30),
        (2560, 720, 60),
        (4416, 1242, 15),
        (3840, 1080, 15),
    ]

    backends = [cv2.CAP_V4L2, None]

    for backend in backends:
        try:
            if backend is None:
                cap = cv2.VideoCapture(device_path)
            else:
                cap = cv2.VideoCapture(device_path, backend)
        except Exception:
            cap = None

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            continue

        try:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            for fourcc in ["YUYV", "MJPG"]:
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                except Exception:
                    pass

                for width, height, fps in test_modes:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, fps)

                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    if actual_h > 0 and actual_w > 0 and actual_w % 2 == 0:
                        eye_w = actual_w // 2
                        if eye_w >= 1000 and actual_h >= 700:
                            return True
        finally:
            cap.release()

    return False


def _download_calibration_file(serial_number):
    serial_number = _normalize_numeric_serial(serial_number)
    if serial_number is None:
        return ""

    candidate_dirs = [
        os.path.join(os.path.expanduser("~"), "zed", "settings"),
        "/usr/local/zed/settings",
    ]

    filename = f"SN{serial_number}.conf"

    for directory in candidate_dirs:
        calibration_file = os.path.join(directory, filename)
        if os.path.isfile(calibration_file):
            return calibration_file

    target_dir = candidate_dirs[0]
    os.makedirs(target_dir, exist_ok=True)
    calibration_file = os.path.join(target_dir, filename)

    url = f"https://calib.stereolabs.com/?SN={serial_number}"
    try:
        urllib.request.urlretrieve(url, calibration_file)
    except Exception:
        return ""

    if not os.path.isfile(calibration_file):
        return ""

    return calibration_file


def _resolution_string_from_width(width):
    if width == 2208:
        return "2K"
    if width == 1920:
        return "FHD"
    if width == 1280:
        return "HD"
    if width == 672:
        return "VGA"

    # Match the official sample fallback behavior.
    return "HD"


def _get_first_float(section, keys, default=0.0):
    for key in keys:
        if key in section:
            try:
                return float(section[key])
            except Exception:
                pass
    return float(default)


def _init_calibration(calibration_file, eye_width, eye_height):
    config = configparser.ConfigParser(interpolation=None)
    config.read(calibration_file)

    if "STEREO" not in config:
        raise RuntimeError(f"Invalid calibration file: missing [STEREO] in {calibration_file}")

    resolution_str = _resolution_string_from_width(eye_width)

    left_section_name = f"LEFT_CAM_{resolution_str}"
    right_section_name = f"RIGHT_CAM_{resolution_str}"

    if left_section_name not in config or right_section_name not in config:
        raise RuntimeError(
            f"Invalid calibration file: missing [{left_section_name}] or [{right_section_name}] in {calibration_file}"
        )

    stereo_section = config["STEREO"]
    left_section = config[left_section_name]
    right_section = config[right_section_name]

    T_ = np.array(
        [
            -_get_first_float(stereo_section, ["Baseline"]),
            _get_first_float(stereo_section, [f"TY_{resolution_str}"]),
            _get_first_float(stereo_section, [f"TZ_{resolution_str}"]),
        ],
        dtype=np.float64,
    )

    left_cam_cx = _get_first_float(left_section, ["cx"])
    left_cam_cy = _get_first_float(left_section, ["cy"])
    left_cam_fx = _get_first_float(left_section, ["fx"])
    left_cam_fy = _get_first_float(left_section, ["fy"])
    left_cam_k1 = _get_first_float(left_section, ["k1"])
    left_cam_k2 = _get_first_float(left_section, ["k2"])
    left_cam_p1 = _get_first_float(left_section, ["p1"])
    left_cam_p2 = _get_first_float(left_section, ["p2"])
    left_cam_k3 = _get_first_float(left_section, ["k3", "p3"])

    right_cam_cx = _get_first_float(right_section, ["cx"])
    right_cam_cy = _get_first_float(right_section, ["cy"])
    right_cam_fx = _get_first_float(right_section, ["fx"])
    right_cam_fy = _get_first_float(right_section, ["fy"])
    right_cam_k1 = _get_first_float(right_section, ["k1"])
    right_cam_k2 = _get_first_float(right_section, ["k2"])
    right_cam_p1 = _get_first_float(right_section, ["p1"])
    right_cam_p2 = _get_first_float(right_section, ["p2"])
    right_cam_k3 = _get_first_float(right_section, ["k3", "p3"])

    # The official sample uses RX / CV / RZ. Some calibration files or older code
    # refer to the middle rotation component differently, so support both.
    R_zed = np.array(
        [
            _get_first_float(stereo_section, [f"RX_{resolution_str}"]),
            _get_first_float(stereo_section, [f"CV_{resolution_str}", f"RY_{resolution_str}"]),
            _get_first_float(stereo_section, [f"RZ_{resolution_str}"]),
        ],
        dtype=np.float64,
    )

    R, _ = cv2.Rodrigues(R_zed)

    raw_left_matrix = np.array(
        [
            [left_cam_fx, 0, left_cam_cx],
            [0, left_cam_fy, left_cam_cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    raw_right_matrix = np.array(
        [
            [right_cam_fx, 0, right_cam_cx],
            [0, right_cam_fy, right_cam_cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    raw_left_dist = np.array(
        [left_cam_k1, left_cam_k2, left_cam_p1, left_cam_p2, left_cam_k3],
        dtype=np.float64,
    )
    raw_right_dist = np.array(
        [right_cam_k1, right_cam_k2, right_cam_p1, right_cam_p2, right_cam_k3],
        dtype=np.float64,
    )

    T = np.array([[T_[0]], [T_[1]], [T_[2]]], dtype=np.float64)

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        cameraMatrix1=raw_left_matrix,
        cameraMatrix2=raw_right_matrix,
        distCoeffs1=raw_left_dist,
        distCoeffs2=raw_right_dist,
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        imageSize=(eye_width, eye_height),
        newImageSize=(eye_width, eye_height),
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        raw_left_matrix,
        raw_left_dist,
        R1,
        P1,
        (eye_width, eye_height),
        cv2.CV_32FC1,
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        raw_right_matrix,
        raw_right_dist,
        R2,
        P2,
        (eye_width, eye_height),
        cv2.CV_32FC1,
    )

    rect_left_matrix = P1[:3, :3]
    rect_right_matrix = P2[:3, :3]

    return (
        raw_left_matrix,
        raw_right_matrix,
        raw_left_dist,
        raw_right_dist,
        rect_left_matrix,
        rect_right_matrix,
        map_left_x,
        map_left_y,
        map_right_x,
        map_right_y,
        P1,
        P2,
    )