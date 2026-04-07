#!/usr/bin/env python3
"""
clawbot_bridge.py
Telegram -> Desktop bridge:
- /start_program : launches a target Python program and keeps it running
- /stop          : stops the target program
- /send <text>   : sends <text> to the program's stdin
- plain text     : also forwarded to stdin (when running)
- stdout/stderr  : forwarded back to Telegram (batched)

SECURITY:
- Only accepts commands from TELEGRAM_ALLOWED_USER_ID
"""

import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# =======================
# CONFIG
# =======================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
ALLOWED_USER_ID = int(os.environ.get("TELEGRAM_ALLOWED_USER_ID", "0"))

# Set the script you actually want to run.
# Tip: keep "-u" for unbuffered output so you see logs immediately.
TARGET_CMD = [sys.executable, "-u", "/path/to/your_program.py"]

# If your target program needs a working directory, set it here:
TARGET_CWD: Optional[str] = None

# If you want to block forwarding until the program prints READY, set this:
REQUIRE_READY = False
READY_MARKER = "READY"

# =======================
# STATE
# =======================
@dataclass
class ProcState:
    proc: Optional[asyncio.subprocess.Process] = None
    stdout_task: Optional[asyncio.Task] = None
    ready: bool = False
    chat_id: Optional[int] = None


STATE = ProcState()


def authorized(update: Update) -> bool:
    user = update.effective_user
    return user is not None and user.id == ALLOWED_USER_ID


async def send_telegram(app: Application, chat_id: int, text: str):
    # Telegram has message length limits; keep it safe.
    if len(text) > 3500:
        text = text[:3500] + "\n…(truncated)"
    await app.bot.send_message(chat_id=chat_id, text=text)


async def pump_stdout(app: Application, chat_id: int):
    """
    Reads stdout from the running process and forwards it to Telegram.
    Batches lines to reduce spam and respects message length.
    """
    assert STATE.proc is not None and STATE.proc.stdout is not None

    buffer: list[str] = []
    char_count = 0

    async def flush():
        nonlocal buffer, char_count
        if buffer:
            await send_telegram(app, chat_id, "\n".join(buffer))
            buffer = []
            char_count = 0

    try:
        while True:
            line = await STATE.proc.stdout.readline()
            if not line:
                break

            text = line.decode(errors="replace").rstrip()
            if not text:
                continue

            # READY detection
            if REQUIRE_READY and (READY_MARKER in text) and not STATE.ready:
                STATE.ready = True
                await send_telegram(app, chat_id, f"[bridge] Detected {READY_MARKER}. You can send input now.")

            # Batch output
            buffer.append(text)
            char_count += len(text) + 1

            # Flush if batch gets big
            if len(buffer) >= 12 or char_count >= 3000:
                await flush()

        await flush()
    except asyncio.CancelledError:
        # Normal on stop
        pass
    except Exception as e:
        await send_telegram(app, chat_id, f"[bridge] stdout pump error: {e}")


async def start_program(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return

    if STATE.proc and STATE.proc.returncode is None:
        await update.message.reply_text("Already running.")
        return

    chat_id = update.effective_chat.id
    STATE.chat_id = chat_id
    STATE.ready = not REQUIRE_READY

    await update.message.reply_text("Starting program...")

    try:
        STATE.proc = await asyncio.create_subprocess_exec(
            *TARGET_CMD,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=TARGET_CWD,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,  # Linux: create new process group
        )
    except FileNotFoundError:
        STATE.proc = None
        await update.message.reply_text("Failed to start: target command not found. Check TARGET_CMD.")
        return
    except Exception as e:
        STATE.proc = None
        await update.message.reply_text(f"Failed to start: {e}")
        return

    STATE.stdout_task = asyncio.create_task(pump_stdout(context.application, chat_id))

    if REQUIRE_READY:
        await update.message.reply_text(
            f"Program started. Waiting for '{READY_MARKER}' before accepting input."
        )
    else:
        await update.message.reply_text("Program started. Send text (or /send <text>) to forward to stdin.")


async def stop_program(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return

    if not STATE.proc or STATE.proc.returncode is not None:
        await update.message.reply_text("Not running.")
        return

    await update.message.reply_text("Stopping program...")

    try:
        # Try graceful terminate
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            os.killpg(os.getpgid(STATE.proc.pid), signal.SIGTERM)
        else:
            STATE.proc.terminate()

        # Wait a bit
        try:
            await asyncio.wait_for(STATE.proc.wait(), timeout=2.5)
        except asyncio.TimeoutError:
            # Force kill
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(STATE.proc.pid), signal.SIGKILL)
            else:
                STATE.proc.kill()
    except Exception as e:
        await update.message.reply_text(f"Stop error: {e}")

    if STATE.stdout_task:
        STATE.stdout_task.cancel()
        STATE.stdout_task = None

    STATE.proc = None
    STATE.ready = False
    await update.message.reply_text("Stopped.")


async def send_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return

    if not STATE.proc or STATE.proc.returncode is not None or not STATE.proc.stdin:
        await update.message.reply_text("Program not running. Use /start_program.")
        return

    if REQUIRE_READY and not STATE.ready:
        await update.message.reply_text("Still initializing (not READY yet).")
        return

    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Usage: /send <text>")
        return

    try:
        STATE.proc.stdin.write((text + "\n").encode())
        await STATE.proc.stdin.drain()
        await update.message.reply_text("Sent.")
    except Exception as e:
        await update.message.reply_text(f"Send failed: {e}")


async def forward_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Forward any plain text message to the running program's stdin."""
    if not authorized(update):
        return

    if not update.message or not update.message.text:
        return

    msg = update.message.text.strip()
    if not msg or msg.startswith("/"):
        return

    if not STATE.proc or STATE.proc.returncode is not None or not STATE.proc.stdin:
        await update.message.reply_text("Not running. Use /start_program.")
        return

    if REQUIRE_READY and not STATE.ready:
        await update.message.reply_text("Still initializing (not READY yet).")
        return

    try:
        STATE.proc.stdin.write((msg + "\n").encode())
        await STATE.proc.stdin.drain()
        # No ack to avoid chat spam
    except Exception as e:
        await update.message.reply_text(f"Forward failed: {e}")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return

    if not STATE.proc:
        await update.message.reply_text("Status: not started.")
        return

    if STATE.proc.returncode is None:
        await update.message.reply_text(f"Status: running (pid={STATE.proc.pid}), ready={STATE.ready}")
    else:
        await update.message.reply_text(f"Status: exited (code={STATE.proc.returncode})")


async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    await update.message.reply_text(f"Your user id: {user.id}")


def main():
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN env var.")
    if ALLOWED_USER_ID == 0:
        print("WARNING: TELEGRAM_ALLOWED_USER_ID is not set. Use /whoami then set it, or this is insecure.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("whoami", whoami_cmd))
    app.add_handler(CommandHandler("start_program", start_program))
    app.add_handler(CommandHandler("start", start_program))   # alias
    app.add_handler(CommandHandler("stop", stop_program))
    app.add_handler(CommandHandler("send", send_input))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, forward_text))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()