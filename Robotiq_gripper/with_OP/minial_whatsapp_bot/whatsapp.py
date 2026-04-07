"""WhatsApp channel via Node.js Baileys bridge.

Architecture:
    Node.js bridge (bridge/) ←WebSocket→ this module ←→ OpenRouterModel

Setup:
    cd bridge && npm install && npm run build
    node dist/index.js          # terminal 1 — scan QR on first run
    python main.py --whatsapp   # terminal 2
"""
import asyncio
import base64
import json
import logging
import os
import re
import signal
import threading
import time
from collections import OrderedDict
from http import server

import cv2
import depthai as dai
import websockets
from dotenv import load_dotenv

from model import OpenRouterModel

BRIDGE_URL = os.getenv("BRIDGE_URL", "ws://127.0.0.1:8765")
BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "")
ALLOW_FROM = set(filter(None, os.getenv("ALLOW_FROM", "").split(",")))  # comma-separated phone numbers

log = logging.getLogger("whatsapp")


STREAM_TRIGGERS = {"stream", "camera", "snapshot", "photo", "pic"}
_OAK_LOCK = threading.Lock()  # serialise all DAI pipeline open/close calls
LIVE_LINK_TRIGGERS = {"live", "livestream", "watch", "feed", "hls", "webrtc", "rtsp"}
AUTO_LIVE_STREAM = os.getenv("AUTO_LIVE_STREAM", "1").strip() not in {"0", "false", "False"}
AUTO_LIVE_HOST = os.getenv("AUTO_LIVE_HOST", "127.0.0.1").strip()
AUTO_LIVE_PORT = int(os.getenv("AUTO_LIVE_PORT", "8008").strip())
AUTO_LIVE_TUNNEL_PROTOCOL = os.getenv("AUTO_LIVE_TUNNEL_PROTOCOL", "http2").strip() or "http2"
OAK_FRAME_WIDTH = int(os.getenv("OAK_FRAME_WIDTH", "640").strip())
OAK_FRAME_HEIGHT = int(os.getenv("OAK_FRAME_HEIGHT", "360").strip())
OAK_STREAM_FPS = max(1, int(os.getenv("OAK_STREAM_FPS", "30").strip()))
OAK_JPEG_QUALITY = min(100, max(10, int(os.getenv("OAK_JPEG_QUALITY", "80").strip())))


class OakMjpegService:
    def __init__(self, host: str = AUTO_LIVE_HOST, port: int = AUTO_LIVE_PORT):
        self.host = host
        self.port = port
        self._latest_jpeg: bytes | None = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._server_thread: threading.Thread | None = None
        self._httpd: server.ThreadingHTTPServer | None = None

    @property
    def local_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._capture_thread and self._capture_thread.is_alive():
            return
        # Clean up any dead-but-not-joined previous run
        if self._capture_thread:
            self._capture_thread.join(timeout=5)
            self._capture_thread = None

        self._stop.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="oak-capture")
        self._capture_thread.start()
        self._httpd = self._make_http_server()
        self._server_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="oak-mjpeg-http")
        self._server_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._capture_thread:
            self._capture_thread.join(timeout=5)
            self._capture_thread = None

    def _capture_loop(self) -> None:
        with _OAK_LOCK:
            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            q = cam.requestOutput((OAK_FRAME_WIDTH, OAK_FRAME_HEIGHT), dai.ImgFrame.Type.BGR888p).createOutputQueue()
            pipeline.start()
            try:
                while not self._stop.is_set() and pipeline.isRunning():
                    frame = q.get().getCvFrame()
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, OAK_JPEG_QUALITY])
                    if ok:
                        with self._latest_lock:
                            self._latest_jpeg = buf.tobytes()
            except Exception as exc:
                log.error("OAK MJPEG capture failed: %s", exc)
            finally:
                pipeline.stop()

    def _make_http_server(self) -> server.ThreadingHTTPServer:
        parent = self

        class Handler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = (
                        "<html><body style='margin:0;background:#111;color:#fff;font-family:sans-serif;'>"
                        "<div style='padding:12px;'>OAK-D Live Stream</div>"
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
                        while not parent._stop.is_set():
                            with parent._latest_lock:
                                frame = parent._latest_jpeg
                            if frame is None:
                                time.sleep(0.05)
                                continue
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            time.sleep(1.0 / OAK_STREAM_FPS)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    return

                self.send_error(404)

            def log_message(self, format, *args):
                return

        return server.ThreadingHTTPServer((self.host, self.port), Handler)


class CloudflaredTunnel:
    URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")

    def __init__(self):
        self._proc: asyncio.subprocess.Process | None = None
        self._url: str | None = None

    async def ensure(self, local_url: str) -> str | None:
        if self._proc and self._proc.returncode is None and self._url:
            return self._url

        self._proc = await asyncio.create_subprocess_exec(
            "cloudflared",
            "tunnel",
            "--url",
            local_url,
            "--protocol",
            AUTO_LIVE_TUNNEL_PROTOCOL,
            "--no-autoupdate",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        assert self._proc.stdout is not None
        deadline = time.time() + 20
        while time.time() < deadline:
            line = await self._proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="ignore")
            match = self.URL_RE.search(text)
            if match:
                self._url = match.group(0)
                await asyncio.sleep(3)  # wait for DNS propagation
                return self._url

        await self.stop()
        return None

    async def stop(self):
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            await self._proc.wait()
        self._proc = None
        self._url = None


def build_live_stream_message() -> str:
    # Reload env so updated .env values can be picked up without process restart.
    load_dotenv(override=True)
    live_stream_url = os.getenv("LIVE_STREAM_URL", "").strip()
    live_stream_hls_url = os.getenv("LIVE_STREAM_HLS_URL", "").strip()
    live_stream_webrtc_url = os.getenv("LIVE_STREAM_WEBRTC_URL", "").strip()
    live_stream_rtsp_url = os.getenv("LIVE_STREAM_RTSP_URL", "").strip()

    lines: list[str] = []
    if live_stream_url:
        lines.append(f"Live stream: {live_stream_url}")
    if live_stream_hls_url:
        lines.append(f"HLS: {live_stream_hls_url}")
    if live_stream_webrtc_url:
        lines.append(f"WebRTC: {live_stream_webrtc_url}")
    if live_stream_rtsp_url:
        lines.append(f"RTSP: {live_stream_rtsp_url}")

    if lines:
        return "Camera live links:\n" + "\n".join(lines)
    return (
        "Live stream URL is not configured.\n"
        "Set LIVE_STREAM_URL or LIVE_STREAM_HLS_URL / LIVE_STREAM_WEBRTC_URL / LIVE_STREAM_RTSP_URL."
    )


def capture_oak_frame() -> bytes:
    """Capture a single JPEG frame from OAK-D RGB camera."""
    with _OAK_LOCK:
        p = dai.Pipeline()
        cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        q = cam.requestOutput((OAK_FRAME_WIDTH, OAK_FRAME_HEIGHT), dai.ImgFrame.Type.BGR888p).createOutputQueue()
        p.start()
        try:
            frame = q.get().getCvFrame()
        finally:
            p.stop()
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, OAK_JPEG_QUALITY])
    return buf.tobytes()


class WhatsAppBot:
    def __init__(self):
        self.model = OpenRouterModel(log_level=logging.INFO)
        self.histories: dict[str, list] = {}          # sender → conversation history
        self.seen: OrderedDict[str, None] = OrderedDict()  # dedup last 1000 message ids
        self._ws = None
        self._mjpeg = OakMjpegService()
        self._tunnel = CloudflaredTunnel()

    async def run(self):
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(self._shutdown()))
        loop.add_signal_handler(signal.SIGINT, lambda: loop.create_task(self._shutdown()))
        log.info("Connecting to bridge at %s", BRIDGE_URL)
        try:
            while True:
                try:
                    async with websockets.connect(BRIDGE_URL) as ws:
                        self._ws = ws
                        if BRIDGE_TOKEN:
                            await ws.send(json.dumps({"type": "auth", "token": BRIDGE_TOKEN}))
                        log.info("Connected to WhatsApp bridge")
                        async for raw in ws:
                            await self._handle(raw)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._ws = None
                    log.warning("Bridge disconnected: %s — retrying in 5s", e)
                    await asyncio.sleep(5)
        finally:
            await self._cleanup()

    async def _shutdown(self):
        log.info("Shutting down...")
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    async def _cleanup(self):
        log.info("Cleaning up streaming resources...")
        self._mjpeg.stop()
        await self._tunnel.stop()

    async def _handle(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type")

        if msg_type == "message":
            await self._on_message(data)
        elif msg_type == "status":
            log.info("WhatsApp status: %s", data.get("status"))
        elif msg_type == "qr":
            log.info("QR code received — scan in the bridge terminal")
        elif msg_type == "error":
            log.error("Bridge error: %s", data.get("error"))

    async def _on_message(self, data: dict):
        msg_id = data.get("id", "")
        if msg_id:
            if msg_id in self.seen:
                return
            self.seen[msg_id] = None
            if len(self.seen) > 1000:
                self.seen.popitem(last=False)

        sender = data.get("sender", "")       # full JID e.g. 1234567890@s.whatsapp.net
        pn = data.get("pn", "")
        chat_id = sender                       # reply to same JID
        phone = (pn or sender).split("@")[0]
        content = data.get("content", "").strip()

        if not content or not chat_id:
            return

        # Allowlist check
        if ALLOW_FROM and phone not in ALLOW_FROM:
            log.info("Ignored message from %s (not in ALLOW_FROM)", phone)
            return

        log.info("Message from %s: %s", phone, content[:80])

        # Explicit command: /live
        if content.lower().startswith("/live"):
            await self._send(chat_id, await self._build_live_or_auto_message())
            return

        # Check for camera trigger words
        words = set(content.lower().split())
        if words & LIVE_LINK_TRIGGERS:
            await self._send(chat_id, await self._build_live_or_auto_message())
            return

        if words & STREAM_TRIGGERS:
            await self._send(chat_id, "Capturing frame from OAK-D...")
            try:
                # If MJPEG is streaming, wait for its cached frame (shares the device)
                if self._mjpeg._capture_thread and self._mjpeg._capture_thread.is_alive():
                    loop = asyncio.get_event_loop()
                    def wait_for_frame():
                        deadline = time.time() + 3
                        while time.time() < deadline:
                            with self._mjpeg._latest_lock:
                                f = self._mjpeg._latest_jpeg
                            if f:
                                return f
                            time.sleep(0.05)
                        raise RuntimeError("Timed out waiting for MJPEG frame")
                    jpeg = await loop.run_in_executor(None, wait_for_frame)
                else:
                    jpeg = await asyncio.get_event_loop().run_in_executor(None, capture_oak_frame)
                await self._send_image(chat_id, jpeg, "OAK-D RGB snapshot")
            except Exception as e:
                log.error("Camera capture failed: %s", e)
                await self._send(chat_id, f"Camera error: {e}")
            return

        hist = self.histories.setdefault(chat_id, [])
        hist.append({"role": "user", "content": content})

        # Run blocking model call in thread pool to keep event loop free
        reply = await asyncio.get_event_loop().run_in_executor(
            None, self.model.generate_response, hist
        )
        hist.append({"role": "assistant", "content": reply})

        await self._send(chat_id, reply)

    async def _send(self, to: str, text: str):
        if not self._ws:
            log.warning("Not connected, cannot send")
            return
        try:
            await self._ws.send(json.dumps({"type": "send", "to": to, "text": text}, ensure_ascii=False))
        except Exception as e:
            log.error("Send failed: %s", e)

    async def _send_image(self, to: str, jpeg: bytes, caption: str = ""):
        if not self._ws:
            log.warning("Not connected, cannot send image")
            return
        try:
            payload = {"type": "send_image", "to": to, "image": base64.b64encode(jpeg).decode(), "caption": caption}
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            log.error("Send image failed: %s", e)

    async def _build_live_or_auto_message(self) -> str:
        configured = build_live_stream_message()
        if "not configured" not in configured:
            return configured

        if not AUTO_LIVE_STREAM:
            return configured

        try:
            try:
                self._mjpeg.start()
            except OSError as exc:
                # If another local stream process already owns the port, reuse it.
                if exc.errno != 98:
                    raise
                log.warning(
                    "Local stream port %s already in use; reusing existing service",
                    self._mjpeg.port,
                )
            public_url = await self._tunnel.ensure(self._mjpeg.local_url)
            if public_url:
                return (
                    "Camera live links:\n"
                    f"Live stream: {public_url}\n"
                    f"MJPEG direct: {public_url}/stream.mjpg"
                )
            return (
                "Live stream auto-start failed: could not create Cloudflare tunnel.\n"
                "Install cloudflared or set LIVE_STREAM_URL manually."
            )
        except Exception as exc:
            log.error("Auto live start failed: %s", exc)
            return (
                f"Live stream auto-start failed: {exc}\n"
                "Set LIVE_STREAM_URL manually if needed."
            )


def run_bot():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(WhatsAppBot().run())
