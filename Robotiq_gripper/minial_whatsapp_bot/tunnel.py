import time
import asyncio
import os

AUTO_LIVE_TUNNEL_PROTOCOL = os.getenv("AUTO_LIVE_TUNNEL_PROTOCOL", "http2").strip() or "http2"

class CloudflaredTunnel:
    import re
    URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")

    def __init__(self):
        self._proc: asyncio.subprocess.Process | None = None
        self._url: str | None = None

    async def ensure(self, local_url: str) -> str | None:
        if self._proc and self._proc.returncode is None and self._url:
            return self._url

        self._proc = await asyncio.create_subprocess_exec(
            "cloudflared", "tunnel", "--url", local_url, "--protocol", 
            AUTO_LIVE_TUNNEL_PROTOCOL, "--no-autoupdate",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )

        assert self._proc.stdout is not None
        deadline = time.time() + 20
        while time.time() < deadline:
            line = await self._proc.stdout.readline()
            if not line: break
            text = line.decode("utf-8", errors="ignore")
            match = self.URL_RE.search(text)
            if match:
                self._url = match.group(0)
                await asyncio.sleep(3) 
                return self._url
        await self.stop()
        return None

    async def stop(self):
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            await self._proc.wait()
        self._proc = None
        self._url = None
