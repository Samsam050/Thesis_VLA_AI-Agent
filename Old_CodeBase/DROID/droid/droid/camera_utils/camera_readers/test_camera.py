import cv2
import depthai as dai
import pyrealsense2 as rs
import numpy as np
import threading
import time


# --- Camera 1: OAK-D (Luxonis) ---

class OakCamera:
    def __init__(self, device_info: dai.DeviceInfo, size=(640, 480), fps=30):
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)

        self.device = dai.Device(device_info)

        self.pipeline = dai.Pipeline(self.device)

        cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
     

        self.q_rgb = cam.requestOutput(size, type=dai.ImgFrame.Type.BGR888p, fps=fps).createOutputQueue()

        # Start pipeline
        
        if hasattr(self.pipeline, "start"):
            self.pipeline.start()
          
        elif hasattr(self.device, "startPipeline"):
            self.device.startPipeline(self.pipeline)
          

        print(f"[OAK-D] Started")


    def start(self):
        self.thread.start()

    def update(self):
        while self.running and self.pipeline.isRunning():
            pkt = self.q_rgb.tryGet()
            if pkt is not None:
                self.frame = pkt.getCvFrame()
            else:
                time.sleep(0.001)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass

# --- Camera 2: Intel RealSense D455 ---
class RealSenseCamera:
    def __init__(self):
        self.frame = None
        self.running = True
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # 640x480 is the safe zone for USB adapters
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def update(self):
        print("[RealSense] Searching...")
        try:
            self.pipeline.start(self.config)
            print("[RealSense] Connected!")
            while self.running:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=500)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        self.frame = np.asanyarray(color_frame.get_data())
                except RuntimeError:
                    # Skip frame if timeout occurs
                    continue
        except Exception as e:
            print(f"[RealSense] Error initializing pipeline: {e}")

    def stop(self):
        self.running = False
        self.thread.join()

# --- Main Logic ---
def main():
    print("Initializing cameras...")
    device_infos = dai.Device.getAllAvailableDevices()
    oak_left  = OakCamera(device_infos[0])
    #oak_right = OakCamera(device_infos[1])

    rs_cam = RealSenseCamera()

    # Start independent threads
    rs_cam.start()
    oak_left.start()
    #oak_right.start()
    time.sleep(2) # Warmup time

    print("Streaming... Press 'q' to quit.")
    
    while True:
        frame_oak = oak_left.frame
        #frame_oak_2 = oak_right.frame
        frame_rs = rs_cam.frame
        
        # Black screen placeholders if a camera is missing/loading
        if frame_oak is None:
            frame_oak = np.zeros((640, 480, 3), dtype=np.uint8)
            cv2.putText(frame_oak, "WAITING FOR OAK...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            """
        if frame_oak_2 is None:
            frame_oak_2 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame_oak, "WAITING FOR OAK...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            """
        if frame_rs is None:
            frame_rs = np.zeros((640, 480, 3), dtype=np.uint8)
            cv2.putText(frame_rs, "WAITING FOR RS...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine side-by-side
        #combined = np.hstack((frame_oak, frame_rs,frame_oak_2))
        combined = np.hstack((frame_oak, frame_rs))
        cv2.imshow("Left: OAK-D | Right: RealSense", combined)

        if cv2.waitKey(1) == ord('q'):
            break

    oak_left.stop()
    #oak_right.stop()
    rs_cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()