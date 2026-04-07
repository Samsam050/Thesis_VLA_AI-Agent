import sys
import time
import subprocess

text = sys.argv[1]

while True:
    subprocess.run([
        "edge-playback",
        "--text", text,
        "--voice", "en-US-EmmaMultilingualNeural"
    ])
    time.sleep(5)