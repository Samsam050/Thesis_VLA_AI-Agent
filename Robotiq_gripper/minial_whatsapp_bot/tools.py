# Define different tools to use.
"""Tool implementations for nanocode"""

import glob as globlib
import os
import re
import subprocess
import time
import os 
from dotenv import load_dotenv
import shlex
from pathlib import Path
import depthai as dai
import pyrealsense2 as rs
import json
import threading
import cv2
import numpy as np

OAK_FRAME_WIDTH = int(os.getenv("OAK_FRAME_WIDTH", "640").strip())
OAK_FRAME_HEIGHT = int(os.getenv("OAK_FRAME_HEIGHT", "360").strip())
OAK_JPEG_QUALITY = min(100, max(10, int(os.getenv("OAK_JPEG_QUALITY", "80").strip())))

_OAK_LOCK = threading.Lock()
# ANSI colors
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"


def read(args):
    """Read file with line numbers"""
    lines = open(args["path"]).readlines()
    offset = args.get("offset", 0)
    limit = args.get("limit", len(lines))
    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))


def write(args):
    """Write content to file"""
    with open(args["path"], "w") as f:
        f.write(args["content"])
    return "ok"


def edit(args):
    """Replace old with new in file"""
    text = open(args["path"]).read()
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    replacement = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    with open(args["path"], "w") as f:
        f.write(replacement)
    return "ok"


def glob(args):
    """Find files by pattern, sorted by mtime"""
    base_path = args.get("path")
    if not base_path:  # This catches both None and "" safely
        base_path = "."
        
    pattern = (base_path + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"

def grep(args):
    """Search files for regex pattern"""
    pattern = re.compile(args["pat"])
    hits = []
    
    base_path = args.get("path")
    if not base_path:  # This catches both None and "" safely
        base_path = "."
        
    for filepath in globlib.glob(base_path + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath), 1):
                if pattern.search(line):
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"

def bash(args):
    """Run shell command"""
    load_dotenv()
    sudo_pass = os.getenv("SUDO_PASSWORD", "")

    cmd = args["cmd"]
    # Explicitly export the password into this specific shell session
    full_cmd = f"export SUDO_PASSWORD='{sudo_pass}' && {cmd}"
    
    proc = subprocess.Popen(
        full_cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    output_lines = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"  │ {line.rstrip()}", flush=True)
                output_lines.append(line)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        output_lines.append("\n(timed out after 30s)")
    return "".join(output_lines).strip() or "(empty)"

def bg_bash(args):
    """Run a shell command in the background and return pid + log path + quick health check."""
    cmd = args["cmd"]

    log_dir = Path("/tmp/robot_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in cmd[:40]).strip("_")
    log_path = log_dir / f"{stamp}_{safe_name}.log"

    wrapped = f"nohup bash -lc {shlex.quote(cmd)} > {shlex.quote(str(log_path))} 2>&1 & echo $!"
    proc = subprocess.Popen(
        wrapped,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    pid = proc.stdout.read().strip()
    time.sleep(2)

    check = subprocess.run(
        ["bash", "-lc", f"ps -p {pid} >/dev/null && echo RUNNING || echo EXITED"],
        capture_output=True,
        text=True,
    )
    status = check.stdout.strip()

    tail = ""
    try:
        if log_path.exists():
            tail = log_path.read_text(errors="ignore")[-1000:]
    except Exception:
        pass

    return (
        f"Background process PID={pid}\n"
        f"Status after 2s: {status}\n"
        f"Log file: {log_path}\n"
        f"Recent log output:\n{tail if tail else '(no output yet)'}"
    )

def capture_oak_frame() -> bytes:
    """Capture a single JPEG frame from OAK-D RGB camera and IMMEDIATELY release the hardware."""
    with _OAK_LOCK:
        p = dai.Pipeline()
        cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        q = cam.requestOutput((OAK_FRAME_WIDTH, OAK_FRAME_HEIGHT), dai.ImgFrame.Type.BGR888p).createOutputQueue()
        p.start()
        try:
            # Grab exactly one frame
            frame = q.get().getCvFrame()
        finally:
            # THIS is the magic line that turns the camera off so the VLA can use it later
            p.stop()
            
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, OAK_JPEG_QUALITY])
    return buf.tobytes()

def capture_realsense_frame() -> bytes:
    """Capture a single JPEG frame from Intel RealSense camera and IMMEDIATELY release the hardware."""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable the color stream (640x480 is standard and fast)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    pipeline.start(config)
    try:
        # RealSense cameras need to drop a few frames to let the auto-exposure settle
        # Otherwise, the first frame is usually pitch black or extremely dark
        for _ in range(5):
            frames = pipeline.wait_for_frames()
            
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Could not grab RealSense frame")
            
        # Convert to numpy array for OpenCV
        color_image = np.asanyarray(color_frame.get_data())
    finally:
        # THIS is the magic line that turns the camera off so the VLA can use it later
        pipeline.stop()
        
    _, buf = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, OAK_JPEG_QUALITY])
    return buf.tobytes()


def get_live_stream_link(args):
    import asyncio
    from tunnel import CloudflaredTunnel
    
    tunnel = CloudflaredTunnel()
    # Force the async function to run synchronously in this thread
    url = asyncio.run(tunnel.ensure("http://127.0.0.1:8008"))
    
    if url:
        return f"Tunnel successful! The live stream URL is: {url}"
    return "Failed to get the tunnel URL after 20 seconds."

def take_snapshot(args):
    """Capture both cameras and save/send-ready images."""
    import base64
    oak_jpeg = capture_oak_frame()
    rs_jpeg = capture_realsense_frame()

    return json.dumps({
        "ok": True,
        "oak_image_b64": base64.b64encode(oak_jpeg).decode(),
        "realsense_image_b64": base64.b64encode(rs_jpeg).decode(),
        "message": "Captured wrist and external camera images."
    })

# Tool definitions: (description, schema, function)
TOOLS = {
    "read": (
        "Read file with line numbers (file path, not directory)",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": (
        "Write content to file",
        {"path": "string", "content": "string"},
        write,
    ),
    "edit": (
        "Replace old with new in file (old must be unique unless all=true)",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex pattern",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "bash": (
        "Run shell command",
        {"cmd": "string"},
        bash,
    ),
    "bg_bash": (
        "Run shell command in the background (non-blocking, use for starting servers or robots)",
        {"cmd": "string"},
        bg_bash,
    ),
    "get_live_stream_link": (
        "Creates a public tunnel to the robot's camera and returns the URL. Use this when the user wants to watch the robot.",
        {},  # No arguments needed!
        get_live_stream_link,
    ),
    "take_snapshot": (
    "Capture a fresh image from the wrist camera and external camera when the user asks to see the scene, asks for an image, photo, snapshot, or asks what the robot sees.",
    {},
    take_snapshot,
),
}


def run_tool(name, args):
    """Execute a tool by name with given arguments"""
    try:
        return TOOLS[name][2](args)
    except Exception as err:
        return f"error: {err}"


def make_schema():
    """Generate Responses-API-compatible tool schema"""
    result = []

    for name, (description, params, _fn) in TOOLS.items():
        properties = {}
        required = []

        for param_name, param_type in params.items():
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")

            if base_type == "number":
                json_type = "integer"
            elif base_type == "boolean":
                json_type = "boolean"
            else:
                json_type = "string"

            if is_optional:
                properties[param_name] = {
                    "type": [json_type, "null"]
                }
            else:
                properties[param_name] = {
                    "type": json_type
                }

            # With strict schema, include every property in required
            required.append(param_name)

        result.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                "strict": True,
            }
        )

    return result