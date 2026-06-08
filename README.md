# franka-gripper Branch

> **Why this branch exists:** The Robotiq gripper (see `robotiq-gripper` branch) occasionally stops mid-task for unknown reasons. This branch was created to isolate and debug the issue using the built-in Franka Hand gripper instead. **No fix was found** — the root cause remains unknown. If you want to continue debugging, this is the right branch to work from.

## Repository Branches

| Branch | Description |
|--------|-------------|
| `fine-tuning` | Fine-tuning scripts and experiments (e.g. episode visualization) |
| `franka-gripper` | **This branch** — Franka Hand gripper, debugging mid-task stop (unresolved) |
| `robotiq-gripper` | Main benchmark branch with Robotiq gripper, VLM agent, and WhatsApp bot |
| `Old_code` | Earlier experiments and initial development code |

---

## Scripts

| Script | Type | OpenAI API Key |
|--------|------|----------------|
| `vla.py` | VLA only | No |

This branch contains only one policy script (`vla.py`). There is no VLM agent and no WhatsApp integration. For those features, see the `robotiq-gripper` branch.

---

## Requirements

- NUC running Ubuntu 18.04 or 22.04 (with RT kernel — see DROID Setup below)
- Franka Emika Panda robot with Franka Hand gripper
- `polymetis-local` conda environment (installed as part of DROID setup)
- A separate machine with a GPU to run the OpenPI policy server
- OAK-D camera (wrist) + Intel RealSense camera (external)
- FCI activated on the robot (via Franka Desk)

---

## DROID Setup (NUC)

All commands run on the NUC only.

There are two options depending on your Ubuntu version. An RT (real-time) kernel is required in either case for libfranka timing.

**Ubuntu 18.04** — Follow the official DROID host installation guide:
https://droid-dataset.github.io/droid/software-setup/host-installation.html

**Ubuntu 22.04 (easier, recommended)** — Use the Docker-based setup:
https://droid-dataset.github.io/droid/software-setup/docker.html#booting-with-ubuntu-2204

Regardless of which path you take, make sure the RT kernel is active before proceeding.

After setup, activate the conda environment used throughout:

```bash
conda activate polymetis-local
```

### Franka Robot Configuration

1. Open Franka Desk in a browser at the robot's IP address
2. Unlock the joints and activate FCI
3. Under end-effector settings, select **"Franka Gripper"**

> Do not upload an external config file — the Franka Hand gripper uses the built-in Franka Desk selection.

---

## OpenPI Setup (GPU Machine)

The policy server runs on a separate machine with a GPU. Clone the OpenPI repo and run the server:

```bash
# Clone OpenPI
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Run the policy server with the pi0.5 DROID checkpoint
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

The server listens on port `8000` by default. Note the IP address of this machine — you will need it when configuring `vla.py`.

See the full OpenPI documentation for installation and dependencies:
https://github.com/Physical-Intelligence/openpi

---

## Camera Setup

This branch uses two cameras:

| Camera | Type | Role | Parameter Key |
|--------|------|------|---------------|
| OAK-D | DepthAI | Wrist (hand) | `wrist_camera_id` / `hand_camera_id` |
| Intel RealSense | RealSense | External | `left_camera_id` / `varied_camera_1_id` |

Serial numbers must be set in **two places**:

**1. `droid/droid/misc/parameters.py`**
```python
hand_camera_id = "YOUR_OAK_D_SERIAL"
varied_camera_1_id = "YOUR_REALSENSE_SERIAL"
```

**2. The `Args` dataclass at the top of `scripts/vla.py`**
```python
wrist_camera_id: str = "YOUR_OAK_D_SERIAL"
left_camera_id: str = "YOUR_REALSENSE_SERIAL"
```

Look up the serial number for each camera using the manufacturer's tools and make sure both locations match.

### Using Different Cameras

- **ZED cameras**: Supported but require code changes in the policy script. Refer to the original OpenPI DROID integration code for reference.
- **Other cameras**: You need to first write a camera reader class for the new camera, then register it in the policy code.

---

## Configuration

Before running, set the OpenPI server IP in `scripts/vla.py`:

```python
@dataclasses.dataclass
class Args:
    # ...camera IDs above...
    remote_host: str = "xxx.xxx.xxx.xxx"  # IP of the machine running the OpenPI server
    remote_port: int = 8000
```

---

## Startup Sequence

All terminals run on the NUC. Activate the `polymetis-local` conda environment in each terminal before running.

```bash
conda activate polymetis-local
```

**Terminal 1 — DROID control server**
```bash
cd Thesis_with_DROID/Franka_gripper/droid
python scripts/server/run_server.py
```

**Terminal 2 — Franka Hand gripper**
```bash
launch_gripper.py gripper=franka_hand +robot_ip=xxx.xxx.xxx.xxx
```

**Terminal 3 — Robot controller**
```bash
launch_robot.py robot_ip=xxx.xxx.xxx.xxx use_gripper=true
```

**Terminal 4 — VLA policy**
```bash
cd Thesis_with_DROID/Franka_gripper/droid/scripts
python vla.py
```

When prompted, enter a natural language instruction for the robot (e.g. `Pick up the banana and place it in the bowl`).

---

## Troubleshooting

**Robot stops mid-task**
This is the known issue this branch was created to investigate. The Franka Hand gripper intermittently stops during task execution. No fix was found. If you make progress, consider opening an issue or PR on the repository.

**libfranka timing errors**
Make sure the RT kernel is active. Without it, libfranka will throw timing-related exceptions.

**Camera not found**
Verify the serial numbers in both `parameters.py` and the `Args` dataclass in `vla.py`. Confirm the camera is physically connected and recognized by the OS before launching the policy.

**Policy server not reachable**
Check that `remote_host` in `vla.py` points to the correct IP of the GPU machine and that port `8000` is not blocked by a firewall.
