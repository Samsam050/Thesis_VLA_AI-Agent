import logging
import time
import json
import base64

import cv2
import numpy as np
import tqdm
import tyro
from openai import OpenAI

#import from main.py
from examples.libero.main import (
    Args,
    _get_libero_env,
    _quat2axisangle,
    LIBERO_ENV_RESOLUTION,
    benchmark,
)
# ---
from openpi_client import websocket_client_policy as _websocket_client_policy

#initialize these in the main() function
VLM_CLIENT = None
OPENPI_CLIENT = None
ENV = None
OBS = None

#tools for VLM
functions = [
    {
        "name": "run_robot_action",
        "description": "Executes a simple, short robot command. Use analyze_scene_for_task first.",
        "parameters": {
            "type": "object",
            "properties": {
                "simple_prompt": {
                    "type": "string",
                    "description": "A simple, direct command for the robot, e.g., 'pick up the red block'.",
                },
            },
            "required": ["simple_prompt"],
        },
    },
    {
        "name": "analyze_scene_for_task",
        "description": "Checks if a task is possible by analyzing the current camera image for the required objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "The user's task, e.g., 'pick up the pencil'.",
                },
            },
            "required": ["task_description"],
        },
    },
]

def analyze_scene_for_task(task_description: str) -> str:
    """
    This is your "Sanity Check" tool (Problem 2).
    It asks a VLM if the task is possible with the *current* image.
    """
    global VLM_CLIENT, OBS
    if VLM_CLIENT is None or OBS is None:
        return json.dumps({"status": "error", "message": "Clients or simulation not ready."})

    logging.info(f"VLM SANITY CHECK: Checking if '{task_description}' is possible...")
    
    # Get the current camera image from the global 'obs'
    image = np.ascontiguousarray(OBS["agentview_image"][::-1, ::-1])
    
    # Convert image to base64
    _, buffer = cv2.imencode(".jpg", image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Create the prompt for the VLM Planner
    vlm_prompt = (
        f"You are a robot helper. Look at this image. "
        f"Based *only* on the objects you see, can the robot do this task: '{task_description}'? "
        f"Answer YES or NO, and if NO, briefly explain why (e.g., 'NO, the object does not exist')."
    )
    
    try:
        response = VLM_CLIENT.responses.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vlm_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        answer = response.choices[0].message.content
        logging.info(f"VLM SANITY CHECK: VLM says: {answer}")
        return json.dumps({"status": "success", "analysis": answer})
    except Exception as e:
        logging.error(f"VLM call failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})

def run_robot_action(simple_prompt: str) -> str:
    """
    This is your "Action Loop" tool.
    It runs the OpenPI model in a loop for one simple task.
    """
    global OPENPI_CLIENT, ENV, OBS
    
    logging.info(f"[Agent is acting... running task: '{simple_prompt}']")
    
    # We can get max_steps from the args, but let's hardcode for simplicity
    max_steps = 300 
    
    for t in tqdm.tqdm(range(max_steps)):
        img = np.ascontiguousarray(OBS["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(OBS["wrist_image"][::-1, ::-1])
        
        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    OBS["robot0_eef_pos"],
                    _quat2axisangle(OBS["robot0_eef_quat"]),
                    OBS["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(simple_prompt),
        }

        try:
            action = OPENPI_CLIENT.infer(element)["actions"]
        except Exception as e:
            logging.error(f"Connection to OpenPI server failed: {e}")
            return json.dumps({"status": "error", "message": f"Connection to OpenPI server failed: {e}"})

        # Execute action and get NEW observation
        OBS, reward, done, info = ENV.step(action[0])

        cv2.imshow("env", img)
        cv2.imshow("Wrist", wrist_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return json.dumps({"status": "error", "message": "User quit."})

        # (Problem 3: We will add the VLM Success Checker here later)
        # For now, we just use the (unreliable) BDDL 'done' flag
        if done:
            logging.info(f"BDDL check says task '{simple_prompt}' is complete!")
            return json.dumps({"status": "success", "message": f"Task '{simple_prompt}' completed (BDDL check)."})
    
    logging.warning(f"Task '{simple_prompt}' timed out.")
    return json.dumps({"status": "success", "message": f"Task '{simple_prompt}' timed out."})


def main_agent(args: Args):
    global VLM_CLIENT, OPENPI_CLIENT, ENV, OBS
    
    # --- 5.1 Setup ---
    logging.info("Initializing VLM and OpenPI clients...")
    try:
        VLM_CLIENT = OpenAI()
        OPENPI_CLIENT =_websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        logging.info(f"Connected to OpenPI server at {args.host}:{args.port}")
    except Exception as e:
        logging.critical(f"Failed to initialize clients: {e}. Check API keys and server connection.")
        return
        
    logging.info("Loading simulation environment...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(0) # Just load a base env
    ENV, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    initial_states = task_suite.get_task_init_states(0)
    OBS = ENV.set_init_state(initial_states[0])
    logging.info("Simulation loaded. Ready for prompts.")
    
    input_list = []
    print("---------------------------------")
    print("Welcome to the VLM Robot Agent. Type 'EXIT' to end.")
    
    while True:
        user_input = input("> DU: ")
        if user_input.lower() == "exit":
            break

        input_list.append({"role": "user"
                           , "content": user_input
                           })
        
        try:
            print("[Agent is thinking... deciding which tool to use]")
            response = VLM_CLIENT.responses.create(
                model="gpt-5-nano-2025-08-07", 
                input=input_list,
                tools=functions,
                tool_choice="auto",
            
            )

            response_message = response.choices[0].message
            input_list.append(response_message) # Add VLM's response to history

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    print(f"[Agent decided to call tool: {function_name}()]")

                    if function_name == "analyze_scene_for_task":
                        function_output = analyze_scene_for_task(args["task_description"])
                    
                    elif function_name == "run_robot_action":
                        function_output = run_robot_action(args["simple_prompt"])
                    
                    else:
                        function_output = json.dumps({"status": "error", "message": f"Unknown function: {function_name}"})

                    # Send the tool's output back to the VLM
                    input_list.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_output,
                        }
                    )
                
                # Let the VLM give a final, natural language response
                print("[Agent is generating final response...]")
                final_response = VLM_CLIENT.responses.create(
                    model="gpt-5-nano-2025-08-07",
                    input=input_list,
                )
                final_answer = final_response.choices[0].message.content
                print(f"> LLM: {final_answer}")
                input_list.append({"role": "assistant", "content": final_answer})

            else:
                # If no tool was called, just print the text response
                print(f"> LLM: {response_message.content}")

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            input_list.pop() # Remove the failed user input

    logging.info("Agent shutting down.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main_agent)
