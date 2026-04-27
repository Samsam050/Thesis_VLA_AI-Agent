import contextlib
import dataclasses
import faulthandler
import os
import signal
import time
import json
import re
from collections import deque
import threading
import subprocess
from http import server
from pathlib import Path

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from typing import Optional
from droid.robot_env import RobotEnv
import tyro
import cv2
import logging
from openai import OpenAI
import base64
import ast
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
os.environ["QT_QPA_PLATFORM"] = "xcb"

DROID_CONTROL_FREQUENCY = 15
STEPS_LOG_FILE = "steps.txt"
VLM_MODEL = "gpt-5-mini"

LIVE_HOST = "127.0.0.1"
LIVE_PORT = 8008
INSTRUCTION_FILE = Path("/tmp/robot_instruction.txt")
TTS_SCRIPT = "/home/frankanuc01/Thesis_H/Thesis/Robotiq_gripper/droid/scripts/TTS.py"
ROBOT_UPDATES_FILE = Path("/tmp/robot_updates.jsonl")
ROBOT_RESULT_FILE = Path("/tmp/robot_result.json")
CONFIRMATION_FILE = Path("/tmp/robot_confirmation.txt")

faulthandler.enable()


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "242422301956"  # e.g., "24259877"
    #left_camera_id: str = "19892622"
    wrist_camera_id: str = "19443010513EA12E00"  # e.g., "13062452"
    #wrist_camera_id: str = "19892622"

    # Policy parameters
    external_camera: Optional[str] = "left"  # which external camera should be fed to the policy, choose from ["left"]

    # Rollout parameters
    max_timesteps: int = 5000
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 4

    # Main completion checker cadence
    replan_steps: int = 125

    # Wrong-object monitor cadence
    anomaly_check_steps: int = 90

    # Delay before second wrong-object confirmation
    wrong_object_confirm_delay_steps: int = 10

    # Retry count per subtask
    max_subtask_retries: int = 3

    # How many consecutive missing-target votes are needed
    missing_target_confirmation_needed: int = 2

    # Stall detection
    stall_window_steps: int = 1000
    stall_distance_threshold: float = 0.01  # meters

    missing_target_first_check_steps: int = 30
    missing_target_confirm_delay_steps: int = 80

    # Live stream parameters
    live_host: str = LIVE_HOST
    live_port: int = LIVE_PORT

    # Remote server parameters
    remote_host: str = "130.243.124.173"  # point this to the IP address of the policy server
    remote_port: int = 8000  # default server port for openpi servers is 8000


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


def initialize_feedback_files():
    ROBOT_UPDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ROBOT_UPDATES_FILE.write_text("", encoding="utf-8")
    ROBOT_RESULT_FILE.write_text("{}", encoding="utf-8")
    CONFIRMATION_FILE.write_text("", encoding="utf-8")


def emit_robot_update(event_type: str, message: str, final: bool = False, **extra):
    payload = {
        "timestamp": time.time(),
        "event_type": event_type,
        "message": message,
    }
    if extra:
        payload.update(extra)

    try:
        with ROBOT_UPDATES_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"Failed to append robot update: {e}")

    if final:
        try:
            ROBOT_RESULT_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logging.error(f"Failed to write robot result: {e}")


def _final_result(status: str, message: str = "", event_type: Optional[str] = None, **extra) -> dict:
    event = event_type or status
    emit_robot_update(event, message, final=True, status=status, **extra)
    result = {"status": status}
    if message:
        result["message"] = message
    if extra:
        result.update(extra)
    return result


def initialize_steps_file(file_path: str = STEPS_LOG_FILE):
    """Create or clear the steps log file for a new task run."""
    with open(file_path, "w") as f:
        f.write("")


def append_completed_step(step_count: int, file_path: str = STEPS_LOG_FILE):
    """Append the completed step count to the file, separated by spaces."""
    with open(file_path, "a") as f:
        f.write(f"{step_count} ")


def _encode_obs_images(curr_obs: dict):
    left_rgb = curr_obs["left_image"]
    wrist_rgb = curr_obs["wrist_image"]

    left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)

    _, buf_ext = cv2.imencode(".jpg", left_bgr)
    _, buf_wrist = cv2.imencode(".jpg", wrist_bgr)

    ext_b64 = base64.b64encode(buf_ext).decode("utf-8")
    wrist_b64 = base64.b64encode(buf_wrist).decode("utf-8")
    return ext_b64, wrist_b64


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_parse_json(text: str) -> dict:
    text = _strip_code_fences(text)
    return json.loads(text)


def _query_vlm_with_images(prompt: str, curr_obs: dict) -> str:
    ext_b64, wrist_b64 = _encode_obs_images(curr_obs)

    response = client.responses.create(
        model=VLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{ext_b64}",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{wrist_b64}",
                    },
                ],
            }
        ],
    )
    return response.output_text.strip()


def _parse_supervisor_response(answer: str) -> dict:
    try:
        data = _safe_parse_json(answer)
    except Exception:
        logging.error(f"Could not parse VLM JSON: {answer}")
        return {
            "decision": "error",
            "message": f"Could not parse VLM JSON: {answer}",
            "raw": answer,
        }

    decision = data.get("decision", "error")
    message = data.get("message", "")
    allowed = {
        "continue",
        "done",
        "reset_and_verify",
        "report_missing_target",
        "already_done",
        "ask_user_confirm",
        "wrong_object_detected",
        "missing_target",
        "retry",
        "complete",
        "incomplete",
    }

    if decision not in allowed:
        return {
            "decision": "error",
            "message": f"Invalid decision: {decision}",
            "raw": answer,
        }

    return {
        "decision": decision,
        "message": message,
        "raw": answer,
    }


def _parse_done_no_response(answer: str) -> dict:
    text = _strip_code_fences(answer).strip().lower()

    if text == "done":
        return {"status": "done", "raw": answer}
    if text == "no":
        return {"status": "no", "raw": answer}

    if "done" in text and "no" not in text:
        return {"status": "done", "raw": answer}

    return {"status": "no", "raw": answer}


def preflight_task_reasoning(task_description: str, curr_obs: dict) -> dict:
    """
    Whole-task check BEFORE creating subtasks.
    """
    try:
        logging.info(f"Preflight whole-task reasoning for: {task_description}")

        prompt = (
            f"You are a high-level visual reasoning module for a Franka robot.\n"
            f"The user asked the robot to do this full task:\n"
            f"'{task_description}'\n\n"

            f"You are shown TWO images of the SAME physical scene from TWO camera views.\n"
            f"Use both views together.\n"
            f"Do NOT count the same object twice.\n\n"

            f"This check happens BEFORE planning subtasks.\n"
            f"Your job is to reason about the WHOLE task and the current scene.\n\n"

            f"Allowed decisions:\n"
            f"- continue\n"
            f"- already_done\n"
            f"- ask_user_confirm\n"
            f"- report_missing_target\n\n"

            f"Meaning:\n"
            f"- continue: the task should proceed normally.\n"
            f"- already_done: the requested task already appears satisfied.\n"
            f"- ask_user_confirm: the scene suggests ambiguity and the user should confirm before continuing.\n"
            f"- report_missing_target: the requested target or required object clearly does not exist in the scene.\n\n"

            f"Important reasoning rules:\n"
            f"- Be conservative.\n"
            f"- If the task still seems possible, return continue.\n"
            f"- If uncertain, return continue.\n"
            f"- If an object could still be inside a box, bowl, or hidden area, return continue.\n"
            f"- If a box is closed by a lid, do NOT conclude missing target.\n"
            f"- If a box is open but the inside is not clearly visible enough, do NOT conclude missing target.\n"
            f"- Only return report_missing_target if the target is clearly absent and there is no plausible hidden place left.\n"
            f"- If a task says 'take out X from box' or 'pick up X from box' and X is already clearly outside the box, prefer ask_user_confirm unless the task is clearly already satisfied.\n"
            f"- If the task says 'put X in Y' and X is already clearly in Y, return already_done.\n"
            f"- If the requested final state already clearly holds, return already_done.\n\n"
            f"- Return already_done if the requested final state already holds, even if the relevant object is no longer visible because the task has effectively been completed.\n"
            f"- For cleanup tasks such as 'clean the table', 'tidy the table', or similar, return already_done if no loose objects that need to be moved, there should always be a container for example a box or bowl, which should not count as a loose object.\n"
            f"- Missing target means the task cannot be completed because a required object is absent; already_done means the requested goal already holds.\n"
            f"- Do NOT return report_missing_target for cleanup tasks just because there are no loose target objects left; that often means the cleanup is already done.\n"
            f"- Return report_missing_target only when the task requires a specific target object that is needed to perform the task, and that target is clearly absent while the task goal is not already satisfied.\n"

            f"Examples:\n"
            f"- Task: 'take out the bear from the box'\n"
            f"  If a bear is already clearly visible on the table outside the box: ask_user_confirm.\n"
            f"- Task: 'pick up bear from box'\n"
            f"  If the bear is already clearly outside the box: ask_user_confirm.\n"
            f"- Task: 'put blue cube in the box'\n"
            f"  If the blue cube is already in the box: already_done.\n"
            f"- Task: 'take out the banana from the box'\n"
            f"  If no banana is visible anywhere, the box is clearly open and inspectable, and there is no plausible hidden place left: report_missing_target.\n\n"
            f"- Task: 'clean the table'\n"
            f"  If the table is already clear of loose objects and the cleanup goal already holds: already_done.\n"
            f"- Task: 'tidy the table'\n"
            f"  If no relevant loose objects remain to be moved: already_done.\n"
            f"- Task: 'put the bear in the box'\n"
            f"  If no bear is visible anywhere and there is no plausible hidden place left, while the bear is not already in the box: report_missing_target.\n"
            f"- Task: 'put the bear in the box'\n"
            f"  If the bear is already clearly in the box: already_done.\n"

            f"Return ONLY valid JSON.\n"
            f"Do not write explanation outside JSON.\n"
            f"Do not use markdown.\n"
            f"Do not use code fences.\n"
            f'Do not write any text before or after the JSON.\n'
            f'Use exactly this format: {{"decision":"continue","message":""}}\n'
        )

        answer = _query_vlm_with_images(prompt, curr_obs)
        result = _parse_supervisor_response(answer)
        logging.info(f"Preflight response: {result}")
        return result

    except Exception as e:
        logging.error(f"Failed preflight task reasoning: {e}")
        return {"decision": "error", "message": str(e)}


def check_wrong_object_interaction(subtask: str, curr_obs: dict) -> dict:
    """
    Checks ONLY whether the robot is manipulating the wrong object.
    """
    try:
        logging.info(f"Wrong-object check for subtask: {subtask}")

        prompt = (
            f"You are a cautious visual checker for possible wrong-object interaction.\n"
            f"The robot is currently executing this subtask:\n"
            f"'{subtask}'\n\n"

            f"You are shown TWO images of the SAME scene from TWO camera views.\n"
            f"Use both views together.\n"
            f"Do NOT count the same object twice.\n\n"

            f"Your ONLY job is to detect whether there is a POSSIBLE wrong-object suspicion.\n\n"

            f"Allowed decisions:\n"
            f"- continue\n"
            f"- wrong_object_detected\n\n"

            f"Rules:\n"
            f"- Return wrong_object_detected ONLY if the robot appears to be clearly holding, lifting, or carrying a different object than the requested target.\n"
            f"- If the robot is empty-handed, return continue.\n"
            f"- If the robot is releasing, has just released, or is still close to the correct target object, return continue.\n"
            f"- Do NOT infer wrong object from proximity alone.\n"
            f"- Do NOT infer wrong object from temporary contact alone.\n"
            f"- Do NOT infer wrong object if the scene is ambiguous.\n"
            f"- If uncertain, return continue.\n\n"

            f"Return ONLY valid JSON.\n"
            f"Do not write explanation.\n"
            f"Do not use markdown.\n"
            f"Do not use code fences.\n"
            f'Do not write any text before or after the JSON.\n'
            f'Use exactly this format: {{"decision":"continue","message":""}}\n'
        )

        answer = _query_vlm_with_images(prompt, curr_obs)
        result = _parse_supervisor_response(answer)

        if result["decision"] not in {"continue", "wrong_object_detected"}:
            result = {"decision": "continue", "message": "", "raw": answer}

        logging.info(f"Wrong-object response: {result}")
        return result

    except Exception as e:
        logging.error(f"Failed wrong-object check: {e}")
        return {"decision": "error", "message": str(e)}

def confirm_wrong_object_interaction(subtask: str, curr_obs: dict, args: Args) -> dict:
    """
    Immediate second confirmation for wrong-object detection.
    """
    try:
        logging.info(f"Wrong-object confirmation for subtask: {subtask}")

        prompt = (
            f"You are a strict visual confirmation checker for wrong-object interaction.\n"
            f"The robot is currently executing this subtask:\n"
            f"'{subtask}'\n\n"

            f"You are shown TWO images of the SAME scene from TWO camera views.\n"
            f"Use both views together.\n"
            f"Do NOT count the same object twice.\n\n"

            f"Your ONLY job is to confirm whether the robot is CLEARLY manipulating the WRONG object.\n\n"

            f"Allowed decisions:\n"
            f"- continue\n"
            f"- wrong_object_detected\n\n"

            f"Confirmation rules:\n"
            f"- Be stricter than the first check.\n"
            f"- Return wrong_object_detected ONLY if a wrong object is clearly held, lifted, or carried by the robot.\n"
            f"- If the robot is empty-handed, return continue.\n"
            f"- If the robot is releasing, has just released, or is still near the correct target object, return continue.\n"
            f"- If the scene is ambiguous, return continue.\n"
            f"- Do NOT infer wrong object from proximity alone.\n"
            f"- Do NOT infer wrong object from placement/release motion of the correct target.\n\n"

            f"Return ONLY valid JSON.\n"
            f"Do not write explanation.\n"
            f"Do not use markdown.\n"
            f"Do not use code fences.\n"
            f'Do not write any text before or after the JSON.\n'
            f'Use exactly this format: {{"decision":"continue","message":""}}\n'
        )

        answer = _query_vlm_with_images(prompt, curr_obs)
        result = _parse_supervisor_response(answer)

        if result["decision"] not in {"continue", "wrong_object_detected"}:
            result = {"decision": "continue", "message": "", "raw": answer}

        logging.info(f"Wrong-object confirmation response: {result}")
        return result

    except Exception as e:
        logging.error(f"Failed wrong-object confirmation: {e}")
        return {"decision": "error", "message": str(e)}


def check_subtask_completion(subtask: str, curr_obs: dict) -> dict:
    """
    Main subtask completion checker.
    Returns ONLY done or no.
    """
    try:
        logging.info(f"Subtask completion check for: {subtask}")

        prompt = (
            f"You are a strict visual progress checker for a Franka robot.\n"
            f"Your job is to decide whether this ONE subtask is complete RIGHT NOW:\n"
            f"'{subtask}'\n\n"

            f"You are shown TWO images of the SAME physical scene from TWO camera views.\n"
            f"Use both views together.\n"
            f"Do NOT count the same object twice.\n\n"

            f"This is a LOCAL completion check for ONE subtask only.\n"
            f"Ignore whole-task success.\n"
            f"Judge ONLY whether the named target object has already reached the named final state.\n\n"

            f"Core decision rule:\n"
            f"- Return 'done' if the target object is clearly in the required final state.\n"
            f"- Return 'done' for container-placement subtasks if the target object is no longer visible on the table, is not visible in any clearly wrong location, the robot is not clearly holding or controlling it, and the destination container could hide it behind its walls or occlusion.\n"
            f"- Return 'no' if the target object is clearly in the wrong place.\n"
            f"- Return 'no' if the robot is clearly still holding or directly controlling the target object.\n"
            f"- Return 'no' if the scene is too ambiguous.\n\n"

            f"Important:\n"
            f"- Judge the FINAL OBJECT STATE, not the robot pose alone.\n"
            f"- The arm being nearby is NOT enough for 'no' by itself.\n"
            f"- If the object has disappeared from the table and is not visible anywhere else, you may conclude 'done' ONLY for subtasks where the destination is a box, bowl, or container that could hide the object.\n\n"

            f"Task-specific rules:\n\n"

            f"1. For 'put lid on table':\n"
            f"- Return 'done' ONLY if the lid is clearly visible resting on the table.\n"
            f"- Return 'no' if the lid is still on or over the box.\n"
            f"- Return 'no' if the lid is not clearly visible on the table.\n"
            f"- Return 'no' if the lid is clearly still controlled by the gripper.\n\n"

            f"2. For 'put lid on box':\n"
            f"- Return 'done' ONLY if the lid is clearly visible covering or resting on the box.\n"
            f"- Return 'no' if the lid is on the table or elsewhere.\n"
            f"- Return 'no' if the lid is not clearly visible on the box.\n"
            f"- Return 'no' if the lid is clearly still controlled by the gripper.\n\n"

            f"3. For 'put X in Y' where Y is a box, bowl, or container:\n"
            f"- Return 'done' if X is clearly visible inside Y.\n"
            f"- Return 'done' if X is not visible on the table, is not visible in any clearly wrong location, and is not clearly being held or controlled by the robot, because the most reasonable conclusion is that X is inside Y even if box walls or angle hide it.\n"
            f"- Return 'no' if X is clearly outside Y.\n"
            f"- Return 'no' if X is clearly still held or controlled by the gripper.\n"
            f"- Return 'no' if the scene is too ambiguous.\n\n"

            f"4. For 'put X on table':\n"
            f"- Return 'done' ONLY if X is clearly visible resting on the table.\n"
            f"- Return 'no' if X is not clearly visible on the table.\n"
            f"- Return 'no' if X is inside a box, bowl, or container.\n"
            f"- Return 'no' if X is clearly still controlled by the gripper.\n\n"

            f"5. For 'pick up X from box':\n"
            f"- Return 'done' ONLY if X is clearly visible outside the box and no longer inside it.\n"
            f"- Return 'no' if X is still inside the box.\n"
            f"- Return 'no' if X is partly inside the box.\n"
            f"- Return 'no' if X is not clearly visible outside the box.\n"
            f"- Return 'no' if X is clearly still controlled by the gripper.\n\n"

            f"6. For other 'put X on Y' subtasks:\n"
            f"- Return 'done' ONLY if X is clearly visible on Y.\n"
            f"- Return 'no' if X is not clearly visible on Y.\n"
            f"- Return 'no' if X is clearly still controlled by the gripper.\n\n"

            f"Hard negative rules:\n"
            f"- Do NOT use disappearance alone for 'put X on table' or other visible-placement tasks.\n"
            f"- For container-placement tasks, disappearance from the table can count as completion if the object is not visible in a wrong place and is not clearly still controlled by the robot.\n"
            f"- Do NOT return 'done' just because the robot moved toward the goal.\n\n"

            f"Answer with ONLY ONE WORD:\n"
            f"done\n"
            f"no\n"
        )

        answer = _query_vlm_with_images(prompt, curr_obs)
        result = _parse_done_no_response(answer)
        logging.info(f"Subtask completion response: {result}")
        return result

    except Exception as e:
        logging.error(f"Failed subtask completion check: {e}")
        return {"status": "no", "raw": str(e)}

def should_enable_runtime_missing_target_check(subtask: str, curr_obs: dict) -> bool:
    try:
        logging.info(f"Deciding whether to enable runtime missing-target check for: {subtask}")

        prompt = (
    f"You are a supervisory reasoning module for a Franka robot.\n"
    f"The robot is about to execute this subtask:\n"
    f"'{subtask}'\n\n"

    f"You are shown TWO images of the SAME physical scene from TWO camera views.\n"
    f"Use both views together.\n"
    f"Do NOT count the same object twice.\n\n"

    f"Your job is to decide whether runtime missing-target monitoring should be enabled.\n\n"

    f"Enable runtime missing-target monitoring ONLY when ALL of the following are true:\n"
    f"1. The target object for this subtask is NOT clearly visible right now.\n"
    f"2. The task could still be valid because the target might be hidden, occluded, or inside a container, for example inside a box..\n"
    f"3. Later robot motion or inspection could reveal whether the target is actually present or missing.\n\n"

    f"Do NOT enable runtime missing-target monitoring if the target object is already clearly visible right now,\n"
    f"even if it will later be placed into a box, bowl, or container.\n\n"
    f"DO have in mind that the task could be already done and the targeted object is inside the destination hidden \n\n"


    f"Examples:\n"
    f"- 'pick up bear from box' and no bear is visible, but it could be inside the box -> enable true\n"
    f"- 'take out banana from bowl' and no banana is visible, but it could be hidden in the bowl -> enable true\n"
    f"- 'put blue train in box' and the blue train is clearly visible on the table -> enable false\n"
    f"- 'put cube on plate' and the cube is clearly visible -> enable false\n\n"

    f"Return ONLY valid JSON in exactly one of these forms:\n"
    f'{{"enable": true, "message": ""}}\n'
    f'{{"enable": false, "message": ""}}\n'
    f"Do not use markdown.\n"
    f"Do not use code fences.\n"
    f"Do not write anything outside the JSON.\n"
)

        answer = _query_vlm_with_images(prompt, curr_obs)
        data = _safe_parse_json(answer)

        enable = bool(data.get("enable", False))
        logging.info(f"Runtime missing-target enabled={enable} for subtask: {subtask}")
        return enable

    except Exception as e:
        logging.error(f"Failed runtime-missing-target enable decision: {e}")
        return False


def check_missing_target_during_execution(subtask: str, curr_obs: dict, args: Args) -> dict:
    """
    Runtime missing-target checker for retrieval/container subtasks.
    This is ONLY used after execution has already started.
    """
    try:
        logging.info(f"Runtime missing-target check for subtask: {subtask}")

        prompt = (
            f"You are a strict visual checker for possible missing target during execution.\n"
            f"The robot is currently executing this subtask:\n"
            f"'{subtask}'\n\n"

            f"You are shown TWO images of the SAME physical scene from TWO camera views.\n"
            f"Use both views together.\n"
            f"Do NOT count the same object twice.\n\n"

            f"Allowed decisions:\n"
            f"- continue\n"
            f"- missing_target\n\n"

            f"Your job is to decide whether the target is NOW genuinely missing for this subtask.\n\n"

            f"Important rules:\n"
            f"- Be conservative.\n"
            f"- If uncertain, return continue.\n"
            f"- If the target could still be hidden by box walls, container walls, camera angle, robot arm, gripper, or occlusion, return continue.\n"
            f"- If the container interior is not yet clearly visible enough, return continue.\n"
            f"- If the robot may not yet have inspected the relevant inside area well enough, return continue.\n"
            f"- Return missing_target ONLY if the relevant container/interior is clearly open and inspectable, and the target is still not there.\n"
            f"- Do NOT judge wrong-object interaction.\n"
            f"- Do NOT judge completion.\n"
            f"- This is ONLY a missing-target check.\n\n"

            f"Examples:\n"
            f"- If the box is open but the inside is still partly hidden by the box walls: continue.\n"
            f"- If the box is open, clearly visible inside, and the target object is not there: missing_target.\n"
            f"- If the lid is still on or partly covering the view: continue.\n\n"

            f"Return ONLY valid JSON.\n"
            f"Do not write explanation.\n"
            f"Do not use markdown.\n"
            f"Do not use code fences.\n"
            f'Do not write any text before or after the JSON.\n'
            f'Use exactly this format: {{"decision":"continue","message":""}}\n'
        )

        answer = _query_vlm_with_images(prompt, curr_obs)
        result = _parse_supervisor_response(answer)

        if result["decision"] not in {"continue", "missing_target"}:
            result = {"decision": "continue", "message": "", "raw": answer}

        logging.info(f"Runtime missing-target response: {result}")
        return result

    except Exception as e:
        logging.error(f"Failed runtime missing-target check: {e}")
        return {"decision": "continue", "message": str(e)}


def final_task_completion_check(task_description: str, curr_obs: dict) -> dict:
    """
    Final whole-task checker after planned subtasks are finished.
    """
    try:
        logging.info(f"Final whole-task completion check for: {task_description}")

        prompt = (
            f"You are a final whole-task visual checker for a Franka robot.\n"
            f"The user originally asked the robot to do this full task:\n"
            f"'{task_description}'\n\n"

            f"You are shown TWO images of the SAME physical scene from TWO camera views.\n"
            f"Use both views together.\n"
            f"Do NOT count the same object twice.\n\n"

            f"Allowed decisions:\n"
            f"- complete\n"
            f"- incomplete\n\n"

            f"Rules:\n"
            f"- Be conservative.\n"
            f"- Return complete only if the requested whole-task goal clearly holds.\n"
            f"- Return incomplete if any relevant part still appears unfinished.\n"
            f"- If uncertain, return incomplete.\n"
            f"- For retrieval tasks like 'pick up bear from box' or 'take out bear from box', return complete if the bear is clearly outside the box.\n"
            f"- For placement tasks like 'put cube in box', return complete only if the cube is clearly in the box.\n"
            f"- For cleanup tasks, return complete only if the requested objects are no longer loose in the wrong place.\n\n"

            f"Return ONLY valid JSON.\n"
            f"Do not write explanation.\n"
            f"Do not use markdown.\n"
            f"Do not use code fences.\n"
            f'Do not write any text before or after the JSON.\n'
            f'Use exactly this format: {{"decision":"complete","message":""}}\n'
        )

        answer = _query_vlm_with_images(prompt, curr_obs)
        result = _parse_supervisor_response(answer)

        if result["decision"] not in {"complete", "incomplete"}:
            result = {"decision": "incomplete", "message": "", "raw": answer}

        logging.info(f"Final whole-task completion response: {result}")
        return result

    except Exception as e:
        logging.error(f"Failed final whole-task completion check: {e}")
        return {"decision": "incomplete", "message": str(e)}


def split_task_into_subtasks(task_description: str, curr_obs: dict) -> dict:
    try:
        logging.info(f"Splitting task into subtasks if needed: '{task_description}'")

        prompt = (
            f"You are a strict high-level planner for the Pi0.5 VLA robot.\n"
            f"The user wants to: '{task_description}'.\n\n"

            f"You are shown TWO images of the SAME table scene from TWO camera views.\n"
            f"They show the SAME physical objects from different angles.\n"
            f"The two images are NOT two separate scenes.\n"
            f"An object that appears in both images is usually the SAME physical object.\n\n"

            f"First, mentally match objects across the two views.\n"
            f"Merge any objects that are the same physical item.\n"
            f"Then plan using the set of UNIQUE physical objects only.\n"
            f"Never count the same physical object twice just because it appears in both views.\n"
            f"If a blue cube appears in both images, and there is no clear evidence of two different blue cubes, treat it as ONE blue cube.\n\n"

            f"You must generate a Python list of atomic subtasks using ONLY these forms:\n"
            f"1. 'put [object] on [object]'\n"
            f"2. 'put [object] in [container]'\n"
            f"3. 'pick up [object] from [container]'\n\n"

            f"Priority rule:\n"
            f"- Task-specific rules below OVERRIDE the general rules when they conflict.\n\n"

            f"Rules:\n"
            f"- Use ONLY objects that are clearly visible in at least one view.\n"
            f"- Do NOT invent objects.\n"
            f"- Count UNIQUE physical objects, not image appearances.\n"
            f"- Only create subtasks for loose objects on the table, EXCEPT for explicit box-removal tasks described below.\n"
            f"- Ignore objects already inside a box, EXCEPT for explicit box-removal tasks described below.\n"
            f"- Ignore objects already inside a bowl.\n"
            f"- Ignore objects outside the table.\n"
            f"- Use simple object names.\n"
            f"- Use 'bear', never 'teddy bear'.\n"
            f"- Treat colour as one colour only: use 'blue', not 'light blue' or 'dark blue'.\n"
            f"- When referring to a bowl in any subtask, ALWAYS include its colour (e.g., 'blue bowl', 'red bowl').\n"
            f"- Each loose object may be moved at most ONCE in the plan.\n"
            f"- Do NOT create two subtasks for the same physical object.\n"
            f"- If uncertain whether two same-looking objects in different views are one object or two, prefer treating them as ONE object unless both are clearly visible at the same time in one view or there is strong evidence they are different objects.\n\n"
            f"- Order the subtasks by estimated distance to the robot and cameras.\n"
            f"- Start with objects that appear closest and easiest to reach.\n"
            f"- Leave objects that appear farther away for later.\n"
            f"- THIS IS IMPORTANT: Always start with the objects which is closet to the robot, so for example if the green cube is closer relative to the Franka robot, you have an external camera where you can roughly see the robot, make it the first subtask compared to the yellow train which is little more far.\n\n"

            f"Cleaning rules:\n"
            f"- If both a plate and a box are visible: fruits go in plate, toys go in box.\n"
            f"- If only a plate is visible: all loose objects go in plate.\n"
            f"- If only a box is visible: all loose objects go in box.\n\n"

            f"Lid rules:\n"
            f"- Only use lid actions if a lid is clearly visible.\n"
            f"- If a box is clearly covered by a lid and objects must go in the box, first output: 'put lid on table'\n"
            f"- After all objects are in the box, if the lid is visible on the table, final output may be: 'put lid on box'\n\n"

            f"Box-removal rules:\n"
            f"- If the user asks to take out, remove, retrieve, or pick up an object from a box, this is a special box-removal task.\n"
            f"- For a box-removal task, objects inside the box are RELEVANT and must NOT be ignored.\n"
            f"- If the box is covered by a visible lid, the FIRST subtask must be: 'put lid on table'\n"
            f"- Then output ONLY the retrieval subtask in this form: 'pick up [object] from box'\n"
            f"- If the target object inside the box is clearly visible, name it normally, e.g. 'pick up bear from box'.\n"
            f"- If the task says 'the object' and the target object inside the box is not yet identifiable, you may use the generic name 'object', e.g. 'pick up object from box'.\n"
            f"- Do NOT add 'put object on table' for box-removal tasks.\n"
            f"- Do NOT add 'put lid on box' for box-removal tasks unless the user explicitly asked for re-closing the box.\n"
            f"- Do NOT move unrelated loose objects on the table during a box-removal task unless the user explicitly asked for that.\n"
            f"- Do NOT remove more than the requested target object(s) from the box.\n\n"

            f"Examples:\n"
            f"User: 'Clean up the table'\n"
            f"Image: pear, banana, strawberry, plate\n"
            f"Output: ['put pear in plate', 'put banana in plate', 'put strawberry in plate']\n\n"

            f"User: 'Take out the bear from the box'\n"
            f"Image: box with lid on top\n"
            f"Output: ['put lid on table', 'pick up bear from box']\n\n"

            f"User: 'Pick up the object from the box'\n"
            f"Image: closed box with lid\n"
            f"Output: ['put lid on table', 'pick up object from box']\n\n"

            f"User: 'Stack the red and blue blocks on the green one'\n"
            f"Image: red block, blue block, green block\n"
            f"Output: ['put blue block on green block', 'put red block on blue block']\n\n"

            f"If there's no object, you return an empty list of strings.\n"
            f"Return ONLY the raw Python list of strings.\n"
        )

        answer = _query_vlm_with_images(prompt, curr_obs)
        subtasks = ast.literal_eval(answer)

        if not isinstance(subtasks, list):
            raise ValueError("Model did not return a list")

        logging.info(f"Subtasks: {subtasks}")
        return {"status": "success", "subtasks": subtasks}

    except Exception as e:
        logging.error(f"Failed to split task: {e}")
        return {"status": "error", "message": str(e)}


def verify_remaining_subtasks(task_description: str, env, args: Args) -> dict:
    try:
        curr_obs = _extract_observation(args, env.get_observation())

        final_check = final_task_completion_check(task_description, curr_obs)
        if final_check["decision"] == "complete":
            logging.info("Whole task is complete.")
            return {"status": "complete", "remaining_subtasks": []}

        result = split_task_into_subtasks(task_description, curr_obs)
        if result["status"] != "success":
            return result

        remaining = result["subtasks"]
        if len(remaining) == 0:
            logging.info("No remaining subtasks found. Re-checking current scene.")

            fallback = preflight_task_reasoning(task_description, curr_obs)
            fallback_decision = fallback.get("decision", "error")
            fallback_message = fallback.get("message", "")

            if fallback_decision == "already_done":
                logging.info("Fallback says task is already done.")
                return {"status": "complete", "remaining_subtasks": []}

            if fallback_decision == "report_missing_target":
                logging.info("Fallback says target is missing.")
                return {
                    "status": "missing_target",
                    "message": fallback_message or "Target object not found in the scene.",
                    "remaining_subtasks": []
                }

            logging.info("Fallback could not confirm done or missing target.")
            return {
                "status": "incomplete",
                "message": fallback_message or "Task is not complete, but no remaining subtasks were identified.",
                "remaining_subtasks": []
            }

        logging.info(f"Remaining subtasks found: {remaining}")
        return {"status": "incomplete", "remaining_subtasks": remaining}

    except Exception as e:
        logging.error(f"Failed to verify remaining task: {e}")
        return {"status": "error", "message": str(e)}


def speaker_prompt(subtask: str) -> str:
    try:
        #logging.info(f"Creating prompt for subtask: {subtask}")

        prompt = (
            f"You are the voice module of a friendly, helpful robot. "
            f"You are about to perform the following robotic command: '{subtask}'.\n\n"
            f"Your task is to translate this raw command into a natural, first-person sentence describing what you are doing right now.\n\n"
            f"### RULES:\n"
            f"1. Speak in the first person using present continuous tense (e.g., 'I am putting...').\n"
            f"2. Make it sound natural and conversational, as if you are talking to a human in the room.\n"
            f"3. Keep it brief. Do not add long explanations.\n"
            f"4. OUTPUT ONLY THE SPOKEN SENTENCE. Do not include quotes, markdown formatting, or introductory filler text.\n\n"
            f"### EXAMPLES:\n"
            f"- 'put yellow train in box' -> I am putting the yellow train in the box.\n"
            f"- 'clean table' -> I am going to clean the table now.\n"
            f"- 'put lid on table' -> I am taking the lid off and placing it on the table.\n"
            f"- 'put bear on plate' -> I am placing the bear on the plate.\n\n"
            f"Now, generate the spoken sentence for: '{subtask}'"
        )

        response = client.responses.create(
            model=VLM_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
        )

        answer = response.output_text.strip()
        #logging.info(f"Prompt response: {answer}")
        return answer

    except Exception as e:
        logging.error(f"Failed to create speaker prompt: {e}")
        return subtask


def _build_policy_request(instruction: str, curr_obs: dict, args: Args):
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            curr_obs[f"{args.external_camera}_image"], 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
        "observation/joint_position": curr_obs["joint_position"],
        "observation/gripper_position": curr_obs["gripper_position"],
        "prompt": instruction,
    }


def _binarize_action(action: np.ndarray) -> np.ndarray:
    if action[-1].item() > 0.5:
        action = np.concatenate([action[:-1], np.ones((1,))])
        print("close")
    else:
        action = np.concatenate([action[:-1], np.zeros((1,))])
        print("open")
    return np.clip(action, -1, 1)


def _log_and_reset(env, completed_step=None):
    if completed_step is not None:
        append_completed_step(completed_step)
        print(f"Wrote completed step {completed_step} to {STEPS_LOG_FILE}")
    print("resetting")
    time.sleep(1.0)
    env.reset()
    time.sleep(1.0)


def _speak_subtask(subtask: str):
    spoken_sentence = speaker_prompt(subtask)
    return subprocess.Popen(
        [
            "python3",
            TTS_SCRIPT,
            spoken_sentence,
        ]
    )


def _terminate_process(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _run_subtask_list(subtasks, env, args: Args, policy_client, live_stream) -> dict:
    speaker_process = None
    try:
        for subtask in subtasks:
            _terminate_process(speaker_process)
            speaker_process = _speak_subtask(subtask)

            subtask_obs = _extract_observation(args, env.get_observation())
            runtime_missing_target_enabled = should_enable_runtime_missing_target_check(subtask, subtask_obs)

            result = run_robot_action(env, subtask, args, policy_client, live_stream,runtime_missing_target_enabled)
            if result["status"] != "success":
                print(f"Subtask '{subtask}' ended with status: {result['status']}")
                return result

            #completed_step = result.get("completed_step")
            #_log_and_reset(env, completed_step)
            print("resetting")
            time.sleep(1.0)
            env.reset()
            time.sleep(1.0)
            

        return {"status": "success"}
    finally:
        _terminate_process(speaker_process)

def wait_for_user_confirmation() -> str:
    print(f"Waiting for confirmation file: {CONFIRMATION_FILE}")
    while True:
        try:
            if CONFIRMATION_FILE.exists():
                text = CONFIRMATION_FILE.read_text(encoding="utf-8").strip().lower()
                if text:
                    CONFIRMATION_FILE.write_text("", encoding="utf-8")
                    return text
        except Exception as e:
            logging.error(f"Confirmation read error: {e}")
        time.sleep(0.5)

def run_robot_action(env, instruction, args, policy_client, live_stream, runtime_missing_target_enabled) -> dict:
    max_steps_per_subtask = args.max_timesteps
    total_steps_across_attempts = 0

    for attempt_idx in range(args.max_subtask_retries + 1):
        #print(f"Starting subtask attempt {attempt_idx + 1}/{args.max_subtask_retries + 1}: {instruction}")

        actions_from_chunk_completed = 0
        pred_action_chunk = None
        step_count = 0
        attempt_outcome = None
        cartesian_history = deque(maxlen=args.stall_window_steps)
        runtime_missing_votes = 0
        runtime_missing_first_vote_step = None
        wrong_object_pending = False
        wrong_object_first_step = None

        # Initial pre-check before motion starts:
        # only checks if subtask is already complete.
        start_obs = _extract_observation(args, env.get_observation())
        if runtime_missing_target_enabled:
            print("enabled missing target function")
        start_completion = check_subtask_completion(instruction, start_obs)
        if start_completion["status"] == "done":
            print(f"START CHECK: subtask already done: {instruction}")
            return {
                "status": "success",
                "completed_step": total_steps_across_attempts,
                "message": "",
            }

        try:
            while step_count < max_steps_per_subtask:
                start_time = time.time()
                print(f"Step: {step_count} / {args.max_timesteps}")

                try:
                    curr_obs = _extract_observation(args, env.get_observation())
                except Exception as e:
                    print(f"Camera Error: {e}")
                    return {"status": "error", "message": str(e)}

                img_left = curr_obs["left_image"]
                img_wrist = curr_obs["wrist_image"]

                display_frame = np.concatenate([img_wrist, img_left], axis=1)
                live_stream.update(display_frame)

                cartesian_history.append(curr_obs["cartesian_position"].copy())

                if len(cartesian_history) == args.stall_window_steps:
                    net_move = np.linalg.norm(cartesian_history[-1] - cartesian_history[0])
                    if net_move < args.stall_distance_threshold:
                        print(
                            f"STALL DETECTED: net_move={net_move:.4f} < {args.stall_distance_threshold}. Reset and retry."
                        )
                        attempt_outcome = "stall"
                        break

                # If there is already a wrong-object suspicion, confirm it AFTER more steps.
                if wrong_object_pending and wrong_object_first_step is not None:
                    if step_count >= wrong_object_first_step + args.wrong_object_confirm_delay_steps:
                        confirm_result = confirm_wrong_object_interaction(instruction, curr_obs, args)
                        confirm_decision = confirm_result.get("decision", "continue")

                        if confirm_decision == "wrong_object_detected":
                            print(f"WRONG OBJECT CONFIRMED for subtask '{instruction}'. Reset and retry.")
                            attempt_outcome = "wrong_object"
                            break

                        print(f"WRONG OBJECT suspicion cleared for subtask '{instruction}'.")
                        wrong_object_pending = False
                        wrong_object_first_step = None

                # Wrong-object checker only
                if (
                    not wrong_object_pending
                    and step_count > 0
                    and step_count % args.anomaly_check_steps == 0
                ):
                    wrong_result = check_wrong_object_interaction(instruction, curr_obs)
                    wrong_decision = wrong_result.get("decision", "continue")

                    if wrong_decision == "wrong_object_detected":
                        print(
                            f"WRONG OBJECT suspicion for subtask '{instruction}'. "
                            f"Will confirm after {args.wrong_object_confirm_delay_steps} steps."
                        )
                        wrong_object_pending = True
                        wrong_object_first_step = step_count

                # Runtime missing-target checker:
                # only for retrieval/container subtasks and only after some motion.
                should_check = False

                if runtime_missing_target_enabled:
                    if runtime_missing_votes == 0:
                        should_check = step_count >= args.missing_target_first_check_steps
                    else:
                        should_check = step_count >= runtime_missing_first_vote_step + args.missing_target_confirm_delay_steps

                if should_check:
                    missing_result = check_missing_target_during_execution(instruction, curr_obs, args)
                    missing_decision = missing_result.get("decision", "continue")

                    if missing_decision == "missing_target":
                        runtime_missing_votes += 1
                        if runtime_missing_votes == 1:
                            runtime_missing_first_vote_step = step_count
                        print(
                            f"RUNTIME MISSING-TARGET vote {runtime_missing_votes}/{args.missing_target_confirmation_needed}"
                        )

                        if runtime_missing_votes >= args.missing_target_confirmation_needed:
                            completed_step = total_steps_across_attempts + step_count
                            append_completed_step(completed_step)
                            print(f"MISSING TARGET confirmed for subtask '{instruction}'.")
                            return {
                                "status": "missing_target",
                                "completed_step": total_steps_across_attempts + step_count,
                                "message": missing_result.get("message", ""),
                            }
                    else:
                        runtime_missing_votes = 0


                # Main supervisor only checks done / no
                if step_count > 0 and step_count % args.replan_steps == 0:
                    completion_result = check_subtask_completion(instruction, curr_obs)
                    if completion_result["status"] == "done":
                        print(f"SUCCESS: subtask '{instruction}' marked complete.")
                        return {
                            "status": "success",
                            "completed_step": total_steps_across_attempts + step_count,
                            "message": "",
                        }

                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    #print(instruction)
                    request_data = _build_policy_request(instruction, curr_obs, args)

                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(request_data)["actions"]

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1
                #print("what get from model:")
                #print(action[-1])

                action = _binarize_action(action)
                env.step(action)
                step_count += 1

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

        except KeyboardInterrupt:
            #print("killed ok")
            return {"status": "stopped_by_user", "completed_step": total_steps_across_attempts + step_count}

        total_steps_across_attempts += step_count

        if attempt_outcome == "wrong_object":
            if attempt_idx < args.max_subtask_retries:
                #_log_and_reset(env, total_steps_across_attempts)
                print("resetting")
                time.sleep(1.0)
                env.reset()
                time.sleep(1.0)
                continue

            print(f"FAILED: Repeated wrong-object behavior for subtask '{instruction}'.")
            return {
                "status": "wrong_object_failure",
                "completed_step": total_steps_across_attempts,
                "message": "Robot repeatedly interacted with the wrong object.",
            }

        if attempt_outcome in {"stall", None} and attempt_idx < args.max_subtask_retries:
            #_log_and_reset(env, total_steps_across_attempts)
            print("resetting")
            time.sleep(1.0)
            env.reset()
            time.sleep(1.0)
            continue

        print(f"TIMEOUT: Subtask '{instruction}' reached max retries/steps.")
        return {
            "status": "timeout",
            "completed_step": total_steps_across_attempts,
            "message": "Subtask reached max retries or steps.",
        }


def execute_task_with_vlm(task_description: str, env, args: Args, policy_client, live_stream) -> dict:
    initialize_steps_file()
    initialize_feedback_files()

    curr_obs = _extract_observation(args, env.get_observation())

    # Whole-task preflight check BEFORE planning
    preflight = preflight_task_reasoning(task_description, curr_obs)
    pre_decision = preflight.get("decision", "error")
    pre_message = preflight.get("message", "")

    if pre_decision == "already_done":
        message = pre_message or "The task already appears to be completed."
        print(f"Preflight: task already done. {message}")
        return _final_result("already_done", message, task=task_description)
    if pre_decision == "report_missing_target":
        message = pre_message or "The requested object does not seem to be available."
        print(f"Preflight: missing target. {message}")
        return _final_result("missing_target", message, task=task_description)

    if pre_decision == "ask_user_confirm":
        message = pre_message or "The scene is ambiguous and needs user confirmation before continuing."
        print(f"Preflight warning: {message}")

        emit_robot_update(
            "needs_user_confirm",
            f"{message} Reply yes to continue or no to cancel.",
            status="needs_user_confirm",
            task=task_description,
        )

        user_answer = wait_for_user_confirmation()

        if user_answer not in {"y", "yes"}:
            return _final_result(
                "cancelled_by_user",
                "Task cancelled by user after confirmation request.",
                task=task_description,
            )

        emit_robot_update(
            "confirmation_received",
            "User confirmed. Continuing task.",
            status="confirmation_received",
            task=task_description,
        )
    emit_robot_update("task_started", f"Started task: {task_description}", task=task_description)

    r = split_task_into_subtasks(task_description, curr_obs)
    if r["status"] != "success":
        message = r.get("message", "Failed to split the task into subtasks.")
        return _final_result("error", message, task=task_description)

    subtasks = r.get("subtasks", [task_description])
    if len(subtasks) == 0:
        return _final_result(
            "missing_target",
            "Target object not found in the scene.",
            task=task_description,
        )
    logging.info(f"Identified {len(subtasks)} subtasks: {subtasks}")

    logging.info(f"Starting execution for task: '{subtasks}'")
    result = _run_subtask_list(subtasks, env, args, policy_client, live_stream)
    if result["status"] != "success":
        status = result.get("status", "error")
        message = result.get("message", "Task execution failed.")
        if status == "wrong_object_failure":
            message = result.get("message") or (
                f"The robot was not able to complete the task because it repeatedly interacted with the wrong object while trying to '{task_description}'."
            )
        elif status == "timeout":
            message = result.get("message") or "The robot could not finish the task in time."
        elif status == "stopped_by_user":
            message = result.get("message") or "The task was stopped by the user."
        return _final_result(status, message, task=task_description)

    verification = verify_remaining_subtasks(task_description, env, args)

    if verification["status"] == "complete":
        return _final_result("success_all", "Task completed successfully.", task=task_description)

    if verification["status"] == "incomplete":
        remaining_subtasks = verification["remaining_subtasks"]
        result = _run_subtask_list(remaining_subtasks, env, args, policy_client, live_stream)
        if result["status"] != "success":
            status = result.get("status", "error")
            message = result.get("message", "Task execution failed.")
            if status == "wrong_object_failure":
                message = result.get("message") or "The robot repeatedly interacted with the wrong object and could not complete the task."
            elif status == "timeout":
                message = result.get("message") or "The robot could not finish the task in time."
            return _final_result(status, message, task=task_description)

        final_verification = verify_remaining_subtasks(task_description, env, args)
        if final_verification["status"] == "complete":
            return _final_result("success_all", "Task completed successfully.", task=task_description)

        if final_verification["status"] == "incomplete":
            remaining = final_verification.get("remaining_subtasks", [])
            return _final_result(
                "incomplete",
                "The robot could not fully complete the task.",
                task=task_description,
                remaining_subtasks=remaining,
            )

        message = final_verification.get("message", "Task verification failed.")
        return _final_result(final_verification.get("status", "error"), message, task=task_description)

    message = verification.get("message", "Task verification failed.")
    return _final_result(verification.get("status", "error"), message, task=task_description)


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


def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left"]
    ), f"Please specify an external camera to use for the policy, choose from ['left'], but got {args.external_camera}"

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    time.sleep(3.0)

    live_stream = LiveStreamServer(host=args.live_host, port=args.live_port)
    live_stream.start()

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    while True:
        try:
            instruction = wait_for_instruction()
            if instruction == "exit":
                print("Restting and exitting")
                #_log_and_reset(env)
                time.sleep(1.0)
                env.reset()
                time.sleep(1.0)
                break

            result = execute_task_with_vlm(instruction, env, args, policy_client, live_stream)

            if result["status"] == "success_all":
                print("Task completed successfully.")
                if result.get("message"):
                    print(result["message"])
            elif result["status"] == "missing_target":
                print(f"Task ended: missing target. {result.get('message', '')}")
            elif result["status"] == "needs_user_confirm":
                print(f"Task needs user confirmation. {result.get('message', '')}")
            elif result["status"] == "wrong_object_failure":
                print(f"Task ended: wrong object failure. {result.get('message', '')}")
            elif result["status"] == "already_done":
                print(f"Task already done. {result.get('message', '')}")
            elif result["status"] == "cancelled_by_user":
                print(f"Task cancelled by user. {result.get('message', '')}")
            else:
                print(f"Task ended with status: {result['status']}")
                if result.get("message"):
                    print(result["message"])

            # Reset for next run
            #_log_and_reset(env)
            print("resetting")
            time.sleep(1.0)
            env.reset()
            time.sleep(1.0)

        except KeyboardInterrupt:
            print("Stopped by user. Resetting before shutdown...")
            try:
                #_log_and_reset(env)
                print("resetting")
                time.sleep(1.0)
                env.reset()
                time.sleep(1.0)
            except Exception as e:
                print(f"Reset failed during KeyboardInterrupt handling: {e}")
            break


def _extract_observation(args: Args, obs_dict):
    image_observations = obs_dict["image"]

    left_image, wrist_image = None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key][::-1, ::-1]

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
