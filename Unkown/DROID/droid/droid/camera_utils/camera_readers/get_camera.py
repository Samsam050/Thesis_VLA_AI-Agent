from copy import deepcopy
import cv2
import numpy as np
import depthai as dai
import pyrealsense2 as rs
from droid.misc.parameters import hand_camera_id, varied_camera_1_id
from droid.camera_utils.info import camera_type_dict, get_camera_type
from droid.misc.time import time_ms
import time 
import threading 

def gather_cameras():
    
    all_cameras = []

    # The "List" of cameras to connect
    target_serials = [hand_camera_id, varied_camera_1_id]

    for serial in target_serials:       

        # LOGIC: If this serial matches the Hand ID, it's the OAK-D
        if serial == hand_camera_id:
            try:
                print(f"Identified as OAK-D (Hand). Connecting...")
                cam = OakCamera(serial)
                all_cameras.append(cam)
                print(f"OAK-D Connected!")
            except Exception as e:
                print(f"FAILED to open OAK-D {serial}: {e}")
        
        # LOGIC: Otherwise, it's the RealSense (Body)
        else:
            try:
                print(f"Identified as RealSense (Body). Connecting...")
                cam = RealSenseCamera(serial)
                all_cameras.append(cam)
                print(f"RealSense Connected!")
            except Exception as e:
                print(f"FAILED to open RealSense {serial}: {e}")

    return all_cameras
"""
def time_ms():
    return time.time() * 1000

class OakCamera:
    def __init__(self, serial_number):
        self.pipeline = None
        self.serial_number = str(serial_number)
        # Initialize with target size (480, 640) to match output
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.is_hand_camera = True
        self.high_res_calibration = False
        self.current_mode = "trajectory"
        self._intrinsics = {}
        self.latency = 0
        self.running = True

        print(f"Opening OAK-D (V3 Native Mode): {self.serial_number}")

        try:
            self.pipeline = dai.Pipeline()
            
            # 1. CREATE CAMERA (Using the Unified V3 Node)
            cam = self.pipeline.create(dai.node.Camera).build()
            
            # 2. ENABLE AUTO-FOCUS (CRITICAL FEATURE)
            # This ensures the banana has sharp edges, helping the model judge height/depth.
            try:
                cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
                cam.initialControl.setAutoExposureEnable()
                print(" -> Auto-Focus Enabled")
            except AttributeError:
                print(" -> Warning: Could not set Auto-Focus on this Camera node.")

            # 3. REQUEST OUTPUT (640x480)
            # Instead of using ImageManip (which crashes), we ask the Camera node
            # to give us 640x480 directly. This uses the hardware scaler.
            stream = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.RGB888p)
            
            # 4. CREATE QUEUE
            # Direct output from the camera stream
            self.q_rgb = stream.createOutputQueue(maxSize=1, blocking=False)

            # Start Pipeline
            self.pipeline.start()
            
            # Dummy intrinsics
            K = np.eye(3)
            self._intrinsics = {
                self.serial_number + "_left": {"cameraMatrix": K, "distCoeffs": np.zeros(5)},
                self.serial_number + "_right": {"cameraMatrix": K, "distCoeffs": np.zeros(5)}
            }

            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            
            print("OAK-D Initialized Successfully!")

        except Exception as e:
            print(f"CRITICAL OAK-D ERROR: {e}")
            import traceback
            traceback.print_exc()
        
    def _update(self):
        while self.running:
            try:
                if self.q_rgb is not None:
                    packet = self.q_rgb.tryGet()
                    if packet:
                        self.frame = packet.getCvFrame()
                time.sleep(0.001) 
            except:
                continue

    def read_camera(self):
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        frame = self.frame

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        now = time_ms()
        timestamp_dict[self.serial_number + "_read_end"] = now
        timestamp_dict[self.serial_number + "_frame_received"] = now
        timestamp_dict[self.serial_number + "_estimated_capture"] = now - self.latency
        
        return {
            "image": {
                self.serial_number + "_left": frame,
                self.serial_number + "_right": frame
            }
        }, timestamp_dict

    def disable_camera(self):
        self.running = False
        # Add cleanup if needed
        pass

    def is_running(self):
        return self.current_mode != "disabled"

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    # DROID compatibility stubs
    def set_reading_parameters(self, image=True, **kwargs): pass
    def set_calibration_mode(self): pass
    def set_trajectory_mode(self): self.current_mode = "trajectory"   
    def enable_advanced_calibration(self): pass
    def disable_advanced_calibration(self): pass
    def start_recording(self, filename): pass
    def stop_recording(self): pass
"""


# OAK-D Camera
class OakCamera:
    def __init__(self, serial_number):
        self.pipeline = None
        self.serial_number = str(serial_number)
        self.frame = np.zeros((500, 500, 3), dtype=np.uint8)
        self.is_hand_camera = True
        self.high_res_calibration = False
        self.current_mode = "trajectory"
        #self._current_params = None
        self._intrinsics = {}
        self.latency = 0
        self.running = True

        print(f"Opening OAK-D: {self.serial_number}")

        try:
            #device_info = dai.DeviceInfo(self.serial_number)
            self.pipeline = dai.Pipeline()
            cam = self.pipeline.create(dai.node.Camera).build()
            
            # stream = cam.requestOutput((480, 640), type=dai.ImgFrame.Type.RGB888p)
            stream = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.RGB888p)
            manip = self.pipeline.create(dai.node.ImageManip)
        
    
            #manip.initialConfig.addCrop(500, 500, 500, 500)            
            #manip.initialConfig.addCrop(0, 0, 500, 500)
            stream.link(manip.inputImage)

            self.q_rgb = manip.out.createOutputQueue(maxSize=1, blocking=False)

            self.pipeline.start()
            
            # Set up dummy intrinsics to satisfy DROID
            K = np.eye(3)
            self._intrinsics = {
                self.serial_number + "_left": {"cameraMatrix": K, "distCoeffs": np.zeros(5)},
                self.serial_number + "_right": {"cameraMatrix": K, "distCoeffs": np.zeros(5)}
            }

            # Start background thread
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            
            print("OAK-D Initialized Successfully!")

        except Exception as e:
            print(f"CRITICAL OAK-D ERROR: {e}")
        
    def _update(self):
        while self.running:
            try:
                packet = self.q_rgb.tryGet()
                if packet:
                    self.frame = packet.getCvFrame()
                time.sleep(0.001) 
            except:
                continue

    def read_camera(self):
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        frame = self.frame

        if frame is None:
            frame = np.zeros((500, 500, 3), dtype=np.uint8)
            
        now = time_ms()
        timestamp_dict[self.serial_number + "_read_end"] = now
        timestamp_dict[self.serial_number + "_frame_received"] = now
        timestamp_dict[self.serial_number + "_estimated_capture"] = now - self.latency
        
        return {
            "image": {
                self.serial_number + "_left": frame,
                self.serial_number + "_right": frame
            }
        }, timestamp_dict

    def disable_camera(self):
        self.running = False
        pass

    def is_running(self):
        return self.current_mode != "disabled"

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    # Dummy to satisfy DROID
    def set_reading_parameters(self, image=True, **kwargs): pass
    def set_calibration_mode(self): pass
    def set_trajectory_mode(self): self.current_mode = "trajectory"   
    def enable_advanced_calibration(self): pass
    def disable_advanced_calibration(self): pass
    def start_recording(self, filename): pass
    def stop_recording(self): pass


# RealSense Camera
class RealSenseCamera:
    def __init__(self, serial_number):
        self.serial_number = str(serial_number)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.is_hand_camera = False
        self.high_res_calibration = False
        self.current_mode = "trajectory"
        self._intrinsics = {}
        self.latency = 0
        #self.frame = None
        self.running = True

        print(f"Opening RealSense: {self.serial_number}")

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            self.pipeline.start(config)

            # 3.Dummy
            K = np.eye(3)
            self._intrinsics = {
                self.serial_number + "_left": {"cameraMatrix": K, "distCoeffs": np.zeros(5)},
                self.serial_number + "_right": {"cameraMatrix": K, "distCoeffs": np.zeros(5)}
            }

            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

        except Exception as e:
            print(f"RealSense Error: {e}")

    def _update(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=500)
                color = frames.get_color_frame()
                if color:
                    self.frame = np.asanyarray(color.get_data()).copy()
            except RuntimeError:
                continue

    def read_camera(self):
            timestamp_dict = {self.serial_number + "_read_start": time_ms()}
            frame = self.frame 
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
            now = time_ms()
            #frame = np.zeros((480, 640, 3), dtype=np.uint8)

            timestamp_dict[self.serial_number + "_read_end"] = now
            timestamp_dict[self.serial_number + "_frame_received"] = now
            timestamp_dict[self.serial_number + "_estimated_capture"] = now - self.latency
            
            return {
                "image": {
                    self.serial_number + "_left": frame,
                    self.serial_number + "_right": frame
                }
            }, timestamp_dict

    def disable_camera(self):
        self.running = False
        self.current_mode = "disabled"
        try: self.pipeline.stop()
        except: pass

    def is_running(self):
        return self.current_mode != "disabled"

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    # Dummy to satisfy DROID
    def set_reading_parameters(self, image=True, **kwargs): pass
    def set_calibration_mode(self): pass
    def set_trajectory_mode(self): pass
    def enable_advanced_calibration(self): pass
    def disable_advanced_calibration(self): pass
    def start_recording(self, filename): pass
    def stop_recording(self): pass


"""
#Trying out new featuers 

class RealSenseCamera:
    def __init__(self, serial_number):
        self.serial_number = str(serial_number)
        self.running = True
        self.current_mode = "trajectory"
        self.latency = 0
        self.high_res_calibration = False


        # Image buffers
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth_frame = None
        self.bg_removed_frame = None

        # Experiment toggle
        self.use_depth_bg_removal = True

        print(f"Opening RealSense: {self.serial_number}")

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)

            # Enable color + depth
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

            profile = self.pipeline.start(config)

            # Depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # Alignment (depth → color)
            self.align = rs.align(rs.stream.color)

            # Background removal parameters
            self.clipping_distance_m = 1.0  # meters
            self.clipping_distance = self.clipping_distance_m / self.depth_scale
            self.grey_color = 153

            # Dummy intrinsics (DROID requirement)
            K = np.eye(3)
            self._intrinsics = {
                self.serial_number + "_left": {
                    "cameraMatrix": K,
                    "distCoeffs": np.zeros(5),
                },
                self.serial_number + "_right": {
                    "cameraMatrix": K,
                    "distCoeffs": np.zeros(5),
                },
            }

            # Start capture thread
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

        except Exception as e:
            print(f"RealSense Error: {e}")

    def _update(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=500)

                # Align depth to color
                aligned_frames = self.align.process(frames)
                depth = aligned_frames.get_depth_frame()
                color = aligned_frames.get_color_frame()

                if not depth or not color:
                    continue

                depth_image = np.asanyarray(depth.get_data())
                color_image = np.asanyarray(color.get_data())

                self.frame = color_image.copy()
                self.depth_frame = depth_image

                if self.use_depth_bg_removal:
                    depth_3d = np.dstack(
                        (depth_image, depth_image, depth_image)
                    )
                    self.bg_removed_frame = np.where(
                        (depth_3d > self.clipping_distance)
                        | (depth_3d <= 0),
                        self.grey_color,
                        color_image,
                    ).astype(np.uint8)

            except RuntimeError:
                continue

    def read_camera(self):
        timestamp_dict = {
            self.serial_number + "_read_start": time_ms()
        }

        if self.use_depth_bg_removal and self.bg_removed_frame is not None:
            frame = self.bg_removed_frame
        else:
            frame = self.frame

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        now = time_ms()
        timestamp_dict[self.serial_number + "_read_end"] = now
        timestamp_dict[self.serial_number + "_frame_received"] = now
        timestamp_dict[self.serial_number + "_estimated_capture"] = (
            now - self.latency
        )

        return {
            "image": {
                self.serial_number + "_left": frame,
                self.serial_number + "_right": frame,
            }
        }, timestamp_dict

    def disable_camera(self):
        self.running = False
        self.current_mode = "disabled"
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def is_running(self):
        return self.current_mode != "disabled"

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    # --------------------------------------------------
    # Optional debug visualization (not used by DROID)
    # --------------------------------------------------
    def show_debug(self):
        if self.depth_frame is None:
            return

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self.depth_frame, alpha=0.03),
            cv2.COLORMAP_JET,
        )

        left = (
            self.bg_removed_frame
            if self.use_depth_bg_removal and self.bg_removed_frame is not None
            else self.frame
        )

        images = np.hstack((left, depth_colormap))
        cv2.imshow("RealSense Debug", images)
        cv2.waitKey(1)

    # --------------------------------------------------
    # Dummy functions to satisfy DROID
    # --------------------------------------------------
    def set_reading_parameters(self, image=True, **kwargs):
        pass

    def set_calibration_mode(self):
        pass

    def set_trajectory_mode(self):
        pass

    def enable_advanced_calibration(self):
        pass

    def disable_advanced_calibration(self):
        pass

    def start_recording(self, filename):
        pass

    def stop_recording(self):
        pass
"""