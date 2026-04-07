import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
import json
import re
from collections import deque

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from typing import Optional, Union
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
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

faulthandler.enable()


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "242422301956"  # e.g., "24259877"
    #left_camera_id: str = "19892622"
    right_camera_id: str = "19443010C14F9D2E00"  # e.g., "24514023"
    wrist_camera_id: str = "19443010513EA12E00"  # e.g., "13062452"
    #wrist_camera_id: str = "19892622"

    # Policy parameters
    external_camera: Optional[str] = "left"  # which external camera should be fed to the policy, choose from ["left", "right"]

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

    # Runtime missing-target monitor cadence
    missing_target_check_steps: int = 30

    # How many consecutive missing-target votes are needed
    missing_target_confirmation_needed: int = 2

    # Stall detection
    stall_window_steps: int = 1000
    stall_distance_threshold: float = 0.01  # meters

    missing_target_first_check_steps: int = 30
    missing_target_confirm_delay_steps: int = 80

    # Remote server parameters
    remote_host: str = "130.243.124.161"  # point this to the IP address of the policy server
    remote_port: int = 8000  # default server port for openpi servers is 8000


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


def _is_box_retrieval_text(text: str) -> bool:
    t = text.lower()
    retrieval_words = ["pick up", "take out", "remove", "retrieve", "get "]
    return "from box" in t and any(word in t for word in retrieval_words)


def preflight_task_reasoning(task_description: str, curr_obs: dict, args: Args) -> dict:
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

            f"Examples:\n"
            f"- Task: 'take out the bear from the box'\n"
            f"  If a bear is already clearly visible on the table outside the box: ask_user_confirm.\n"
            f"- Task: 'pick up bear from box'\n"
            f"  If the bear is already clearly outside the box: ask_user_confirm.\n"
            f"- Task: 'put the cube in the box'\n"
            f"  If the cube is already in the box: already_done.\n"
            f"- Task: 'take out the banana from the box'\n"
            f"  If no banana is visible anywhere, the box is clearly open and inspectable, and there is no plausible hidden place left: report_missing_target.\n\n"

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


def check_wrong_object_interaction(subtask: str, curr_obs: dict, args: Args) -> dict:
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
            f"- Be cautious.\n"
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


def check_subtask_completion(subtask: str, curr_obs: dict, args: Args) -> dict:
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


def verify_missing_target_after_attempts(subtask: str, curr_obs: dict, args: Args) -> dict:
    """
    Missing-target check only AFTER the robot already attempted the subtask.
    """
    try:
        logging.info(f"Missing-target verification after attempts for: {subtask}")

        prompt = (
            f"You are a visual verifier for possible missing targets.\n"
            f"The robot has ALREADY attempted this subtask and has now returned to reset pose:\n"
            f"'{subtask}'\n\n"

            f"You are shown TWO images of the SAME scene from TWO camera views.\n"
            f"Use both views together.\n"
            f"Do NOT count the same object twice.\n\n"

            f"Allowed decisions:\n"
            f"- retry\n"
            f"- missing_target\n\n"

            f"Rules:\n"
            f"- Be conservative.\n"
            f"- Return missing_target ONLY if there is strong visual evidence that the target is genuinely unavailable.\n"
            f"- If the target could still be hidden, occluded, or inside a container that is not clearly inspectable, return retry.\n"
            f"- If a box is open but the inside is not clearly visible enough, return retry.\n"
            f"- If box walls, angle, gripper, or occlusion could still hide the target, return retry.\n"
            f"- For 'pick up X from box' or 'take out X from box', return missing_target only if the box is clearly open and inspectable and X is still not there.\n"
            f"- If uncertain, return retry.\n\n"

            f"Return ONLY valid JSON.\n"
            f"Do not write explanation.\n"
            f"Do not use markdown.\n"
            f"Do not use code fences.\n"
            f'Do not write any text before or after the JSON.\n'
            f'Use exactly this format: {{"decision":"retry","message":""}}\n'
        )

        answer = _query_vlm_with_images(prompt, curr_obs)
        result = _parse_supervisor_response(answer)

        if result["decision"] not in {"retry", "missing_target"}:
            result = {"decision": "retry", "message": "", "raw": answer}

        logging.info(f"Missing-target verification response: {result}")
        return result

    except Exception as e:
        logging.error(f"Failed missing-target verification: {e}")
        return {"decision": "retry", "message": str(e)}


def _needs_runtime_missing_target_check(subtask: str) -> bool:
    text = subtask.lower()
    return (
        "from box" in text
        or "from container" in text
        or "from bowl" in text
    )


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


def final_task_completion_check(task_description: str, curr_obs: dict, args: Args) -> dict:
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

        ext_b64, wrist_b64 = _encode_obs_images(curr_obs)

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
            f"- If two objects seem similar in distance, prefer the one that is more clearly visible and less occluded first.\n\n"

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

        answer = response.output_text.strip()
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

        final_check = final_task_completion_check(task_description, curr_obs, args)
        if final_check["decision"] == "complete":
            logging.info("Whole task is complete.")
            return {"status": "complete", "remaining_subtasks": []}

        result = split_task_into_subtasks(task_description, curr_obs)
        if result["status"] != "success":
            return result

        remaining = result["subtasks"]
        if len(remaining) == 0:
            logging.info("Task appears complete after replanning.")
            return {"status": "complete", "remaining_subtasks": []}

        logging.info(f"Remaining subtasks found: {remaining}")
        return {"status": "incomplete", "remaining_subtasks": remaining}

    except Exception as e:
        logging.error(f"Failed to verify remaining task: {e}")
        return {"status": "error", "message": str(e)}


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


def run_robot_action(env, instruction, args, policy_client, display):
    max_steps_per_subtask = args.max_timesteps
    total_steps_across_attempts = 0
    runtime_missing_target_enabled = _needs_runtime_missing_target_check(instruction)

    for attempt_idx in range(args.max_subtask_retries + 1):
        print(f"Starting subtask attempt {attempt_idx + 1}/{args.max_subtask_retries + 1}: {instruction}")

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
        start_completion = check_subtask_completion(instruction, start_obs, args)
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

                display.show("External View", img_left)
                display.show("Wrist View", img_wrist)

                cartesian_history.append(curr_obs["cartesian_position"].copy())

                if len(cartesian_history) == args.stall_window_steps:
                    net_move = np.linalg.norm(cartesian_history[-1] - cartesian_history[0])
                    if net_move < args.stall_distance_threshold:
                        print(
                            f"STALL DETECTED: net_move={net_move:.4f} < {args.stall_distance_threshold}. Reset and retry."
                        )
                        attempt_outcome = "stall"
                        break

                # If there is already a wrong-object suspicion, confirm it AFTER 14 more steps.
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
                    wrong_result = check_wrong_object_interaction(instruction, curr_obs, args)
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
                    completion_result = check_subtask_completion(instruction, curr_obs, args)
                    if completion_result["status"] == "done":
                        print(f"SUCCESS: subtask '{instruction}' marked complete.")
                        return {
                            "status": "success",
                            "completed_step": total_steps_across_attempts + step_count,
                            "message": "",
                        }

                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    print(instruction)
                    request_data = _build_policy_request(instruction, curr_obs, args)

                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(request_data)["actions"]

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1
                print("what get from model:")
                print(action[-1])

                action = _binarize_action(action)
                env.step(action)
                step_count += 1

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

        except KeyboardInterrupt:
            print("killed ok")
            return {"status": "stopped_by_user", "completed_step": total_steps_across_attempts + step_count}

        total_steps_across_attempts += step_count

        if attempt_outcome == "wrong_object":
            if attempt_idx < args.max_subtask_retries:
                append_completed_step(total_steps_across_attempts)
                print(f"Wrote completed step {total_steps_across_attempts} to {STEPS_LOG_FILE}")
                time.sleep(1.0)
                print("resetting")
                env.reset()
                time.sleep(1.0)
                continue

            print(f"FAILED: Repeated wrong-object behavior for subtask '{instruction}'.")
            return {
                "status": "wrong_object_failure",
                "completed_step": total_steps_across_attempts,
                "message": "Robot repeatedly interacted with the wrong object."
            }

        if attempt_outcome in {"stall", None} and attempt_idx < args.max_subtask_retries:
            append_completed_step(total_steps_across_attempts)
            print(f"Wrote completed step {total_steps_across_attempts} to {STEPS_LOG_FILE}")
            time.sleep(1.0)
            print("resetting")
            env.reset()
            time.sleep(1.0)
            continue

        print(f"TIMEOUT: Subtask '{instruction}' reached max retries/steps.")
        return {
            "status": "timeout",
            "completed_step": total_steps_across_attempts,
            "message": "Subtask reached max retries or steps."
        }

    print(f"TIMEOUT: Subtask '{instruction}' reached max retries/steps.")
    return {
        "status": "timeout",
        "completed_step": total_steps_across_attempts,
        "message": "Subtask reached max retries or steps."
    }


def execute_task_with_vlm(task_description: str, env, args: Args, policy_client, display) -> dict:
    initialize_steps_file()
    curr_obs = _extract_observation(args, env.get_observation())

    # Whole-task preflight check BEFORE planning
    preflight = preflight_task_reasoning(task_description, curr_obs, args)
    pre_decision = preflight.get("decision", "error")
    pre_message = preflight.get("message", "")

    if pre_decision == "already_done":
        print(f"Preflight: task already done. {pre_message}")
        return {"status": "success_all", "message": pre_message}

    if pre_decision == "report_missing_target":
        print(f"Preflight: missing target. {pre_message}")
        return {"status": "missing_target", "message": pre_message}

    if pre_decision == "ask_user_confirm":
        print(f"Preflight warning: {pre_message}")
        user_answer = input("Continue anyway? [y/N]: ").strip().lower()
        if user_answer not in {"y", "yes"}:
            return {"status": "cancelled_by_user", "message": pre_message}

    r = split_task_into_subtasks(task_description, curr_obs)
    subtasks = r.get("subtasks", [task_description])
    logging.info(f"Identified {len(subtasks)} subtasks: {subtasks}")

    logging.info(f"Starting execution for task: '{subtasks}'")
    for subtask in subtasks:
        result = run_robot_action(env, subtask, args, policy_client, display)
        if result["status"] != "success":
            print(f"Subtask '{subtask}' ended with status: {result['status']}")
            return result

        completed_step = result.get("completed_step")
        if completed_step is not None:
            append_completed_step(completed_step)
            print(f"Wrote completed step {completed_step} to {STEPS_LOG_FILE}")

        print("resetting")
        time.sleep(1.0)
        env.reset()
        time.sleep(1.0)

    verification = verify_remaining_subtasks(task_description, env, args)

    if verification["status"] == "complete":
        return {"status": "success_all"}

    if verification["status"] == "incomplete":
        remaining_subtasks = verification["remaining_subtasks"]

        for subtask in remaining_subtasks:
            result = run_robot_action(env, subtask, args, policy_client, display)
            if result["status"] != "success":
                print(f"Subtask '{subtask}' ended with status: {result['status']}")
                return result

            completed_step = result.get("completed_step")
            if completed_step is not None:
                append_completed_step(completed_step)
                print(f"Wrote completed step {completed_step} to {STEPS_LOG_FILE}")

            print("resetting")
            time.sleep(1.0)
            env.reset()
            time.sleep(1.0)

        final_verification = verify_remaining_subtasks(task_description, env, args)
        if final_verification["status"] == "complete":
            return {"status": "success_all"}

        return final_verification

    return verification


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


class Display:
    def __init__(self):
        cv2.namedWindow("External View", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Wrist View", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("External View", 960, 540)
        cv2.resizeWindow("Wrist View", 960, 540)

    def show(self, window_name, frame):
        if frame is not None:
            bgr_frame = frame[..., ::-1]   # RGB -> BGR
            cv2.imshow(window_name, bgr_frame)
            cv2.waitKey(1)


def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")
    time.sleep(3.0)
    display = Display()

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    while True:
        try:
            instruction = input("Enter instruction: ")
            if instruction == "exit":
                break

            result = execute_task_with_vlm(instruction, env, args, policy_client, display)

            if result["status"] == "success_all":
                print("Task completed successfully.")
                if result.get("message"):
                    print(result["message"])
            elif result["status"] == "missing_target":
                print(f"Task ended: missing target. {result.get('message', '')}")
            elif result["status"] == "cancelled_by_user":
                print(f"Task cancelled by user. {result.get('message', '')}")
            elif result["status"] == "wrong_object_failure":
                print(f"Task ended: wrong object failure. {result.get('message', '')}")
            else:
                print(f"Task ended with status: {result['status']}")
                if result.get("message"):
                    print(result["message"])

            # Reset for next run
            print("resetting")
            env.reset()

        except KeyboardInterrupt:
            print("Stopped by user.")
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