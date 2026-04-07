# my_bot

A minimal agentic WhatsApp bot.

## Setup

**Prerequisites:** Python 3.12+, Node.js 20+

```bash
# Python deps
uv sync

# Node bridge
cd bridge && npm install && npm run build
```

Create `.env` in the project root:

```env
OPENROUTER_API_KEY=sk-or-...

# Optional
BRIDGE_TOKEN=      # shared secret; leave empty to disable auth
ALLOW_FROM=        # comma-separated phone number allowlist; empty = allow all
LIVE_STREAM_URL=   # generic stream page URL
LIVE_STREAM_HLS_URL=
LIVE_STREAM_WEBRTC_URL=
LIVE_STREAM_RTSP_URL=
```

## Run

**Terminal 1 — WhatsApp bridge:**

```bash
cd bridge && node dist/index.js
```

Scan the QR code in WhatsApp → Linked Devices on first run. Auth is saved to `bridge/wa_auth/`.

**Terminal 2 — bot:**

```bash
uv run python main.py --whatsapp
```

CLI mode (no WhatsApp):

```bash
uv run python main.py
```

## `/live` command

Send `/live` in WhatsApp chat with the bot. It will reply with any configured stream links from:

- `LIVE_STREAM_URL`
- `LIVE_STREAM_HLS_URL`
- `LIVE_STREAM_WEBRTC_URL`
- `LIVE_STREAM_RTSP_URL`

Example:

```env
LIVE_STREAM_URL=https://cam.example.com/live
LIVE_STREAM_HLS_URL=https://cam.example.com/hls/stream.m3u8
LIVE_STREAM_WEBRTC_URL=https://cam.example.com/webrtc
LIVE_STREAM_RTSP_URL=rtsp://cam.example.com:8554/live
```
