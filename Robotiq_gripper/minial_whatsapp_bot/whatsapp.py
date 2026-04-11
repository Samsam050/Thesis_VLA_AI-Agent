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
import signal

from collections import OrderedDict
from pathlib import Path

import websockets
from dotenv import load_dotenv

from model import OpenAIModel
from tunnel import CloudflaredTunnel

BRIDGE_URL = os.getenv("BRIDGE_URL", "ws://127.0.0.1:8765")
BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "")
ALLOW_FROM = set(filter(None, os.getenv("ALLOW_FROM", "").split(",")))

log = logging.getLogger("whatsapp")


class WhatsAppBot:
    def __init__(self):
        self.model = OpenAIModel(log_level=logging.INFO)
        self.histories: dict[str, list] = {}
        self.seen: OrderedDict[str, None] = OrderedDict()
        self._ws = None
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
        await self._tunnel.stop()

    async def _handle(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type")
        if msg_type == "message":
            await self._on_message(data)
        elif msg_type == "error":
            log.error("Bridge error: %s", data.get("error"))

    async def _on_message(self, data: dict):
        msg_id = data.get("id", "")
        if msg_id:
            if msg_id in self.seen: return
            self.seen[msg_id] = None
            if len(self.seen) > 1000: self.seen.popitem(last=False)

        sender = data.get("sender", "")
        pn = data.get("pn", "")
        chat_id = sender
        phone = (pn or sender).split("@")[0]
        content = data.get("content", "").strip()

        if not content or not chat_id:
            return

        # 1. Require the @bot prefix to wake up
        if not content.lower().startswith("@bot"):
            return
            
        # Strip "@bot" out so the AI/logic just sees the raw command
        clean_content = content[4:].strip()

        if ALLOW_FROM and phone not in ALLOW_FROM:
            return

        log.info("Message from %s: %s", phone, clean_content[:80])

        # 3. Pass everything else to the AI Agent
        hist = self.histories.setdefault(chat_id, [])
        hist.append({"role": "user", "content": clean_content})

        reply = await asyncio.get_event_loop().run_in_executor(
            None, self.model.generate_response, hist
        )
        hist.append({"role": "assistant", "content": reply})
        # If the model returned snapshot JSON, send images instead of plain text
        try:
            parsed = json.loads(reply)
            if isinstance(parsed, dict) and parsed.get("ok") and "oak_image_b64" in parsed and "realsense_image_b64" in parsed:
                await self._send(chat_id, "Taking a quick look through both cameras...")
                await self._send_image(chat_id, base64.b64decode(parsed["oak_image_b64"]), "Wrist Camera (OAK-D)")
                await self._send_image(chat_id, base64.b64decode(parsed["realsense_image_b64"]), "External Camera (RealSense)")
                hist.append({"role": "assistant", "content": "[Sent snapshot images]"})
                return
        except Exception:
            pass

        await self._send(chat_id, reply)

    async def _send(self, to: str, text: str):
        if not self._ws: return
        try:
            await self._ws.send(json.dumps({"type": "send", "to": to, "text": text}, ensure_ascii=False))
        except Exception as e:
            log.error("Send failed: %s", e)

    async def _send_image(self, to: str, jpeg: bytes, caption: str = ""):
        if not self._ws: return
        try:
            payload = {"type": "send_image", "to": to, "image": base64.b64encode(jpeg).decode(), "caption": caption}
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            log.error("Send image failed: %s", e)


def run_bot():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(WhatsAppBot().run())