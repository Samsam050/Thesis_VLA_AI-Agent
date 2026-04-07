from pathlib import Path
import sys

INSTRUCTION_FILE = Path("/tmp/robot_instruction.txt")

text = " ".join(sys.argv[1:]).strip()
if not text:
    raise SystemExit("Usage: python send_instruction.py 'clean table'")

INSTRUCTION_FILE.write_text(text, encoding="utf-8")
print(f"Sent instruction: {text}")