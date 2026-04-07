import contextlib
import dataclasses
import faulthandler
import os
import signal
import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from typing import Optional
from droid.robot_env import RobotEnv
import tqdm
import tyro
import cv2
import os
import subprocess
import threading
from http import server
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "xcb"

DROID_CONTROL_FREQUENCY = 15
LIVE_HOST = "127.0.0.1"
LIVE_PORT = 8008
INSTRUCTION_FILE = Path("/tmp/robot_instruction.txt")

faulthandler.enable()

@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "242422301956"  # e.g., "24259877"
    #right_camera_id: str = "19443010C14F9D2E00"  # e.g., "24514023"
    wrist_camera_id: str = "19443010513EA12E00"  # e.g., "13062452"

    # Policy parameters
    external_camera: Optional[str] = "left" # which external camera should be fed to the policy, choose from ["left", "right"]
    
    # Rollout parameters
    max_timesteps: int = 5000
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "130.243.124.161"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt

class LiveStreamServer:
    def __init__(self, host=LIVE_HOST, port=LIVE_PORT, fps=15, jpeg_quality=80):
        self.host = host
        self.port = port
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._latest_jpeg = None
        self._lock = threading.Lock()
        self._httpd = None
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        parent = self

        class Handler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = (
                        "<html><body style='margin:0;background:#111;color:#fff;font-family:sans-serif;'>"
                        "<div style='padding:12px;'>Robot View</div>"
                        "<img src='/stream.mjpg' style='width:100%;height:auto;display:block'/>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()

                    try:
                        while True:
                            with parent._lock:
                                frame = parent._latest_jpeg

                            if frame is None:
                                time.sleep(0.05)
                                continue
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            time.sleep(1.0 / parent.fps)
                    except (BrokenPipeError, ConnectionResetError):
                        return

                self.send_error(404)

            def log_message(self, format, *args):
                return

        self._httpd = server.ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="run-policy-mjpeg")
        self._thread.start()
        print(f"Live stream server running at http://{self.host}:{self.port}")

    def update(self, rgb_frame):
        if rgb_frame is None:
            return
        bgr = rgb_frame[..., ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

def wait_for_instruction():
    print(f"Waiting for instruction file: {INSTRUCTION_FILE}")
    while True:
        try:
            if INSTRUCTION_FILE.exists():
                text = INSTRUCTION_FILE.read_text(encoding="utf-8").strip()
                if text:
                    INSTRUCTION_FILE.write_text("", encoding="utf-8")
                    print(f"Received instruction: {text}")
                    return text
        except Exception as e:
            print(f"Instruction read error: {e}")
        time.sleep(0.5)


def main(args: Args):
    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    time.sleep(3.0)
    live_stream = LiveStreamServer()
    live_stream.start()
    speaker_process = None
    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    while True:
        #instruction = input("Enter instruction: ")
        instruction = wait_for_instruction()

        if speaker_process is not None:
            speaker_process.terminate()
            speaker_process.wait()
        #TODO: ADD LLM with this in the VLM_policy so it sounds more naturally speaking what doing instead of saying just prompt
        speaker_process = subprocess.Popen([
            "python3",
            "/home/frankanuc01/Thesis_H/new_try/droid/scripts/TTS.py",
            instruction
        ])

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        bar = tqdm.tqdm(range(args.max_timesteps))
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                )

                #img_right = curr_obs["right_image"]
                img_left = curr_obs[f"{args.external_camera}_image"]
                img_wrist = curr_obs["wrist_image"]
        
                display_frame = np.concatenate([img_wrist, img_left],axis=1)
                live_stream.update(display_frame)


                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    #"observation/exterior_image_2_left": image_tools.resize_with_pad(curr_obs["right_image"], 224, 224),
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs[f"{args.external_camera}_image"], 224, 224),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (15, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
            
                actions_from_chunk_completed += 1

                print(action[-1])
                if action[-1].item() > 0.5:
                    #Closes gripper
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    #Opens gripper
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)
               

                env.step(action)
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break
        

def _extract_observation(args: Args, obs_dict,):
    image_observations = obs_dict["image"]
    
    left_image, wrist_image = None, None
    for key in image_observations:
        if args.left_camera_id in key:
            left_image = image_observations[key]
        elif args.wrist_camera_id in key:
            wrist_image = image_observations[key]
        

    if left_image is None or wrist_image is None:
         raise ValueError("Critical Error: Left or Wrist camera missing!")

    # Drop alpha & Convert BGR to RGB
    left_image = left_image[..., :3][..., ::-1]   
    wrist_image = wrist_image[..., :3][..., ::-1]
    robot_state = obs_dict["robot_state"]

    return {
        "left_image": left_image, 
        "wrist_image": wrist_image,
        "cartesian_position": np.array(robot_state["cartesian_position"]),
        "joint_position": np.array(robot_state["joint_positions"]),
        "gripper_position": np.array([robot_state["gripper_position"]]),
    }

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
