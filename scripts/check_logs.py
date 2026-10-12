"""Print the latest tqdm status line of every logs/*.log."""
import re
from pathlib import Path

for f in sorted(Path("logs").glob("*.log")):
    txt = f.read_text(encoding="utf-8", errors="replace")
    parts = [p for p in re.split(r"[\r\n]", txt) if p.strip()]
    tq = [p for p in parts if "iterations" in p]
    last = tq[-1] if tq else (parts[-1] if parts else "")
    err = [p for p in parts if "Error" in p or "Traceback" in p]
    ev = [p for p in parts if "eval" in p.lower() and "return" in p.lower()]
    print(f"== {f.stem}: {last.strip()[-160:]}")
    if err:
        print("   ERR:", err[-1][-200:])
    if ev:
        print("   EVAL:", ev[-1][-200:])
