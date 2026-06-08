# Thesis with DROID — `robotiq-gripper` branch

This README covers the **robotiq-gripper** branch, which is the main working branch of this thesis. It integrates the [DROID](https://droid-dataset.github.io/) robot framework with the [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) from [OpenPI](https://github.com/Physical-Intelligence/openpi) on a Franka Emika Panda robot with a Robotiq gripper. It optionally adds an OpenAI-powered AI agent and WhatsApp control.

Each branch has its own README. See the branch table below for what each branch covers.

---

## Repository Branches

| Branch | Description |
|--------|-------------|
| `fine-tuning` | Fine-tuning scripts and experiments |
| `franka-gripper` | Same setup as this branch but for the Franka Hand gripper. Created to debug a mid-task stopping issue (see [Troubleshooting](#troubleshooting)). |
| `robotiq-gripper` | **This branch.** Full robot setup with the Robotiq gripper, VLA, AI agent, and WhatsApp integration. |
| `Old_code` | Earlier experiments and initial code before the current setup was established. |

---

## Scripts

All runnable scripts are in `scripts/`. Each targets a specific combination of features:

| Script | VLA (π₀.₅) | AI Agent (VLM) | WhatsApp | OpenAI key needed |
|--------|:----------:|:---------------:|:--------:|:-----------------:|
| `run_policy.py` | ✅ | ❌ | ❌ | No |
| `run_policy_bot.py` | ✅ | ❌ | ✅ | Yes¹ |
| `VLM_policy.py` | ✅ | ✅ | ❌ | Yes |
| `VLM_policy_bot.py` | ✅ | ✅ | ✅ | Yes |
| `VLM_thinking_policy.py` | ✅ | ✅ (chain-of-thought) | ❌ | Yes |

¹ `run_policy_bot.py` itself does not call OpenAI directly, but it is designed to be used with the WhatsApp bot (`minial_whatsapp_bot/`), which does require an OpenAI key. It also starts a local camera stream that the WhatsApp bot exposes via a Cloudflare tunnel — `cloudflared` must be installed on the machine.

### How each script receives a task instruction

| Script | How to give it a task |
|--------|-----------------------|
| `run_policy.py` | Prompts `Enter instruction:` in the terminal |
| `VLM_policy.py` | Prompts `Enter instruction:` in the terminal |
| `VLM_thinking_policy.py` | Prompts `Enter instruction:` in the terminal |
| `run_policy_bot.py` | Reads from `/tmp/robot_instruction.txt`, written by the WhatsApp bot |
| `VLM_policy_bot.py` | Reads from `/tmp/robot_instruction.txt`, written by the WhatsApp bot |

### API keys

The AI agent scripts use OpenAI as the VLM. An `OPENAI_API_KEY` is required. Create a `.env` file in `scripts/`:

```env
OPENAI_API_KEY=sk-...
```

The model used is `gpt-5-mini` (configurable via `VLM_MODEL` at the top of each script). To switch to a different provider, change the `OpenAI()` client initialisation and `VLM_MODEL` in the script — refer to the OpenAI Python client docs for other compatible endpoints.

---

## Cameras

This branch uses:

| Role | Camera | Serial number field |
|------|--------|---------------------|
| Wrist / hand | **OAK-D** | `wrist_camera_id` / `hand_camera_id` |
| External (scene) | **Intel RealSense** | `left_camera_id` / `varied_camera_1_id` |

The policy uses one external camera and the wrist camera. The `external_camera` argument in each script selects which external camera (`"left"` or `"right"`).

Serial numbers are set in two places and must match:

**`droid/misc/parameters.py`:**
```python
hand_camera_id     = "..."   # OAK-D serial
varied_camera_1_id = "..."   # RealSense serial
varied_camera_2_id = "..."   # second external (if used)
```

**`Args` dataclass at the top of each policy script:**
```python
left_camera_id:  str = "..."   # must match varied_camera_1_id
wrist_camera_id: str = "..."   # must match hand_camera_id
```

Look up the serial number for each camera and make sure it matches in both places.

### Using other cameras

**ZED cameras** — can be used but require changes to the policy script to handle the ZED image format. Refer to the original [DROID examples in the OpenPI repo](https://github.com/Physical-Intelligence/openpi/tree/main/examples/droid) as a starting point.

**Other cameras** — require two steps: (1) write a camera reader class for the new camera, and (2) update the policy script to load and feed frames from it. Use the existing `OakCamera` and `RealSenseCamera` classes in `droid/camera_utils/camera_readers/get_camera.py` as reference.

---

## Hardware Requirements

GPU memory requirements for the OpenPI policy server:

This branch uses inference only, so > 8 GB VRAM is sufficient (e.g. RTX 4090). For fine-tuning requirements see the `fine-tuning` branch.

---

## 1. DROID Setup (First Time)

This is a one-time hardware and software setup. All robot scripts run on the NUC. Full documentation: [DROID Host Installation](https://droid-dataset.github.io/droid/software-setup/host-installation.html).

### 1.1 Franka Robot

Activate FCI (Franka Control Interface) on the robot via Franka Desk. FCI must be active before any robot scripts can run.

**Robotiq gripper:** Upload the end-effector config in Franka Desk under *Settings > End-Effector*: [endeffector-config.json](https://github.com/frankaemika/external_gripper_example/blob/master/panda_with_robotiq_gripper_example/config/endeffector-config.json)

**Franka gripper:** Select `franka gripper` in Franka Desk under *Settings > End-Effector*.

### 1.2 NUC Setup

The NUC runs all robot scripts and the Polymetis controller server. A real-time kernel is required regardless of Ubuntu version — without it, libfranka will not meet timing requirements.

Depending on which Ubuntu version you use, there are two setup paths:

**Option A — Ubuntu 18.04** (original DROID setup): follow the full host installation guide: [DROID Host Installation — NUC section](https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-the-nuc). This covers the static IP, RT kernel patch, CPU frequency scaling, and Conda + Polymetis build.

**Option B — Ubuntu 22.04** (may be easier to get running): follow the Docker-based guide: [DROID Docker Setup](https://droid-dataset.github.io/droid/software-setup/docker.html#booting-with-ubuntu-2204). You still need to apply the RT kernel patch.

> **Note:** If you use Miniconda instead of Anaconda, change `anaconda` to `miniconda` in `droid/franka/launch_gripper.sh`, `launch_robot.sh`, and `scripts/server/launch_server.sh`, and use absolute paths.

After the Polymetis build, install remaining dependencies with the `polymetis-local` conda environment active:

```bash
pip install -e .
pip install dm-robotics-moma==0.5.0 --no-deps
pip install dm-robotics-transformations==0.5.0 --no-deps
pip install dm-robotics-agentflow==0.5.0 --no-deps
pip install dm-robotics-geometry==0.5.0 --no-deps
pip install dm-robotics-manipulation==0.5.0 --no-deps
pip install dm-robotics-controllers==0.5.0 --no-deps
```

Update `droid/misc/parameters.py`:

```python
robot_ip           = "xxx.xxx.xxx.xxx"  # Franka control box IP
nuc_ip             = "xxx.xxx.xxx.xxx"  # NUC IP
sudo_password      = "..."
robot_type         = "panda"            # or "fr3"
robot_serial_number = "..."
hand_camera_id     = "..."              # OAK-D serial
varied_camera_1_id = "..."             # RealSense serial
```

---

## 2. OpenPI Setup (π₀.₅ Policy Server)

The policy server runs on a desktop with a GPU and streams actions to the NUC over WebSocket. Repository: [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

### 2.1 Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2.2 Start the Policy Server

This project uses the **π₀.₅-DROID** checkpoint, which is downloaded automatically on first run.

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

The server listens on port `8000`.

### 2.3 Policy Server IP

Each policy script connects to the server via WebSocket. The IP is set in the `Args` dataclass near the top of each script:

```python
remote_host: str = "xxx.xxx.xxx.xxx"  # IP of the desktop running the policy server
remote_port: int = 8000
```

Update `remote_host` to the IP of the machine running the OpenPI server.

---

## 3. Startup Sequence

### Prerequisites

Before starting, make sure:
- The OpenPI policy server is running on the GPU desktop (Section 2.2)
- The robot is powered on and FCI is active in Franka Desk
- The correct end-effector is set in Franka Desk (Section 1.1)

---

All commands below run on the NUC. Open a new terminal for each step.

### Option A — Robotiq Gripper

**Terminal 1:** Start the Polymetis control server

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate polymetis-local
python /path/to/droid/scripts/server/run_server.py
```

**Terminal 2:** Start the Robotiq gripper

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate polymetis-local
bash /path/to/droid/droid/franka/launch_gripper.sh
```

**Terminal 3:** Start the Franka robot controller

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate polymetis-local
bash /path/to/droid/droid/franka/launch_robot.sh
```

**Terminal 4:** Run the policy script

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate polymetis-local
cd /path/to/droid/scripts
python run_policy.py          # VLA only — type task in terminal
python VLM_policy.py          # AI agent — type task in terminal
python VLM_thinking_policy.py # AI agent with chain-of-thought — type task in terminal
python run_policy_bot.py      # VLA + WhatsApp
python VLM_policy_bot.py      # AI agent + WhatsApp
```

---

### Option B — Franka Gripper

Same as Option A, except Terminal 2 uses a different command:

**Terminal 2:** Start the Franka Hand gripper

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate polymetis-local
cd /path/to/droid/droid/franka
python launch_gripper.py gripper=franka_hand +robot_ip=xxx.xxx.xxx.xxx
```

Terminals 1, 3, and 4 are identical to Option A.

> **Note:** The Franka gripper has a known issue where it may stop unexpectedly mid-task. See [Troubleshooting](#troubleshooting). If you want to investigate this, switch to the `franka-gripper` branch which has a separate README with notes on the debugging attempts.

---

## 4. WhatsApp Integration

The WhatsApp setup consists of two components running on the NUC:

1. **Policy script** (`run_policy_bot.py` or `VLM_policy_bot.py`) — runs the robot and starts a local MJPEG camera stream on port `8008`
2. **WhatsApp bot** (`minial_whatsapp_bot/`) — receives WhatsApp messages, writes task instructions to `/tmp/robot_instruction.txt`, and creates a public Cloudflare tunnel to the camera stream

The bot uses OpenAI (`gpt-5-nano`) as its AI agent. It reads its system prompt from `minial_whatsapp_bot/CLAUDE.md`.

### 4.1 Requirements

- Python 3.12+, Node.js 20+
- `cloudflared` installed and available in PATH ([install guide](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/))
- OpenAI API key

### 4.2 Setup

```bash
cd minial_whatsapp_bot

# Python dependencies
uv sync

# Node.js WhatsApp bridge
cd bridge && npm install && npm run build
```

Create `.env` in `minial_whatsapp_bot/`:

```env
OPENAI_API_KEY=sk-...

# Optional
BRIDGE_TOKEN=        # shared secret; leave empty to disable auth
ALLOW_FROM=          # comma-separated phone number allowlist; empty = allow all
LIVE_STREAM_URL=     # fallback stream URL if tunnel is not used
```

### 4.3 Run

First start the policy script (Terminal 4 from the startup sequence above), then:

**Terminal 5:** WhatsApp bridge
```bash
cd minial_whatsapp_bot/bridge && node dist/index.js
```
On first run, scan the QR code in WhatsApp → **Linked Devices**. Auth is saved to `bridge/wa_auth/` for future runs.

**Terminal 6:** WhatsApp bot
```bash
cd minial_whatsapp_bot
uv run python main.py --whatsapp
```

### 4.4 CLI mode (no WhatsApp)

```bash
uv run python main.py
```

---

## Troubleshooting

**Franka gripper stops mid-task** — The Franka Hand gripper may stop unexpectedly during task execution. The `franka-gripper` branch was created to investigate and debug this issue, but no resolution has been found. It is recommended to use the Robotiq gripper for reliable operation. If you want to continue debugging the Franka issue, switch to the `franka-gripper` branch and refer to its README.

**Cameras not activating (WhatsApp bot only)** — `run_policy_bot.py` and `VLM_policy_bot.py` wait for a non-empty `/tmp/robot_instruction.txt` before activating cameras. This file is written by the WhatsApp bot. It is not relevant when using the standalone terminal scripts.

**Policy server not reachable** — Make sure the OpenPI server is running and that `remote_host` in the policy script matches the server's IP. Default port is `8000`.

**Gripper not responding** — Verify the correct end-effector config is set in Franka Desk before starting the gripper scripts.

**Conda not found** — Source manually: `source ~/miniconda3/etc/profile.d/conda.sh`

**`uv sync` fails** — Run `rm -rf .venv && uv sync`. Also try `uv self update`.

**Cloudflare tunnel not starting** — Verify `cloudflared` is installed and accessible: `cloudflared --version`. The tunnel may also fail if port `8008` is not reachable locally — confirm `run_policy_bot.py` is running first.

---

## References

- [DROID Dataset & Framework](https://droid-dataset.github.io/)
- [DROID Host Installation Guide](https://droid-dataset.github.io/droid/software-setup/host-installation.html)
- [DROID Docker Setup (Ubuntu 22.04)](https://droid-dataset.github.io/droid/software-setup/docker.html#booting-with-ubuntu-2204)
- [OpenPI Repository](https://github.com/Physical-Intelligence/openpi)
- [π₀.₅ Model](https://www.physicalintelligence.company/blog/pi05)
- [Polymetis](https://facebookresearch.github.io/fairo/polymetis/overview.html)
- [Franka FCI Setup](https://frankaemika.github.io/docs/getting_started.html#preparing-the-robot-for-fci-usage-in-desk)
