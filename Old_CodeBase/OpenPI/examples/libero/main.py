import collections
import dataclasses
import logging
import math
import pathlib
import cv2
import imageio
import third_party.libero.libero.libero.benchmark as benchmark
from third_party.libero.libero.libero import get_libero_path
from third_party.libero.libero.libero.envs import SegmentationRenderEnv
#from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "130.243.124.173"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #######################################Ì£##########################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )#check out libero_suite_task.py in benchmark folder
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    #print(num_tasks_in_suite)
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 270  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    # calling the model
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in [0]:
    #tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        #print(initial_states)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        print("here")
        custom_prompt = input("Enter your prompt: ")
        
        #print(task_description)
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {custom_prompt}")

            # Reset environment
            env.reset()
           
            action_plan = collections.deque()
            print(action_plan)
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img         = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img   = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    bird_img    = np.ascontiguousarray(obs["birdview_image"][::-1, ::-1])
                    side_img    = np.ascontiguousarray(obs["sideview_image"][::-1, ::-1])
                    gallery_img = np.ascontiguousarray(obs["galleryview_image"][::-1, ::-1])
                    robot_img   = np.ascontiguousarray(obs["robot0_robotview_image"][::-1, ::-1])
                    front_img   = np.ascontiguousarray(obs["frontview_image"][::-1, ::-1])
                    #print("segmentation_mask" in obs)
                    #print(obs)
                    #print(obs.keys()) 
                    #print("FEL")
                    # Print out the contents and check the type
                    print("Segmentation Mask Shape:", obs["robot0_eye_in_hand_segmentation_instance"].shape)
                    print("Segmentation Mask Dtype:", obs["robot0_eye_in_hand_segmentation_instance"].dtype)

                    segmentation_instance = obs["robot0_eye_in_hand_segmentation_instance"]
                    segmentation_instance = segmentation_instance.squeeze()
                    segmentation_instance = segmentation_instance.astype(np.uint8)

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    bird_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(bird_img, args.resize_size, args.resize_size)
                    )
                    side_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(side_img, args.resize_size, args.resize_size)
                    )
                    gallery_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(gallery_img, args.resize_size, args.resize_size)
                    )
                    robot_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(robot_img, args.resize_size, args.resize_size)
                    )
                    front_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(front_img, args.resize_size, args.resize_size)
                    )
                    segmentation_mask = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(segmentation_instance, args.resize_size, args.resize_size)
                    )
                    print(f"After resizing - Shape: {segmentation_mask.shape}, Dtype: {segmentation_mask.dtype}")
                    if len(segmentation_mask.shape) == 3:
                        segmentation_mask = segmentation_mask[0]
                    
                    print(f"Final Shape: {segmentation_mask.shape}, Dtype: {segmentation_mask.dtype}")
                    print(f"Segmentation Mask Min: {segmentation_mask.min()}")
                    print(f"Segmentation Mask Max: {segmentation_mask.max()}")
                    segmentation_color = cv2.applyColorMap(segmentation_mask.astype(np.uint8), cv2.COLORMAP_JET)
                    #print(obs.keys())

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    #block of code is constructing the observation that will be fed into the model.
                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        print(obs["robot0_gripper_qpos"]) #is two array with numbers likke [0.03872 -0.038]
                        # the numbers stands for distance from the center in meters. so the right of the center and the left of the center

                        print(obs["robot0_eef_pos"]) # XYZ cartesin coordinates

                        print(_quat2axisangle(obs["robot0_eef_quat"]))
                        obs = env.env._get_observations()
                        print("Available Keys:", obs.keys())
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/segmentation_mask": segmentation_mask,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"], # 
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(custom_prompt),
                        }
                        #print(f"Observation element: {element}")
                        # Query model to get action from the model
                        action_chunk = client.infer(element)["actions"]
                        print(action_chunk)
                        # Print the predicted actions
                        #print(f"Predicted actions: {action_chunk}")

                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    
                    cv2.imshow("Wrist", wrist_img) #added this so i am able to view what the robot does.
                    cv2.imshow("env", img)
                    cv2.imshow("bird", bird_img)
                    cv2.imshow("gallery", gallery_img)
                    cv2.imshow("robot", robot_img)
                    cv2.imshow("front", front_img)
                    cv2.imshow("side", side_img)
                    cv2.imshow("Segmentation", segmentation_color)


                    key = cv2.waitKey(1)

                    if key == ord('q'):
                        logging.info("exit")
                        break

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

# initialization_noise = {"type":"gaussian","magnitude": 0.05} #this is done so the robot doesnt start in the exact same position every time, adding some jitter. where magnitude is the intensity of the jitter
# this noise is needed when using data collection or like testing the robustness, so like evaulating to see the sucess rate. or when collecting data so that ai learns to adjust "if a am slightly to the left move right etc"
# other noises that can be added is like observation noise for the sensor, action noise for the motor. the type could also be uniform instead of gaussian. 
def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    print(f"Task Description: {task_description}")

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, }
    env = SegmentationRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)

