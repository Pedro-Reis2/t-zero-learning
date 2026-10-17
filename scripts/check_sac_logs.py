"""Inspect progress of all logs_sac/*.log files."""
import re
from pathlib import Path

for f in sorted(Path("logs_sac").glob("*.log")):
    txt = f.read_text(encoding="utf-8", errors="replace")
    lines = [p.strip() for p in re.split(r"[\r\n]", txt) if p.strip()]
    prog = [p for p in lines if "SAC steps:" in p or "PPO iterations:" in p]
    last_prog = prog[-1] if prog else (lines[-1] if lines else "")
    eval_line = [p for p in lines if "eval/mean_return=" in p or "eval_episode=9" in p]
    errs = [p for p in lines if "Traceback" in p or "Error:" in p]
    # Replace unicode block chars with '#' for safe printing on any console encoding
    safe_prog = re.sub(r"[^\x00-\x7f]", "#", last_prog)
    print(f"== {f.stem:18s}: {safe_prog[-115:]}")
    if eval_line:
        print(f"   EVAL: {eval_line[-1]}")
    if errs:
        print(f"   ERR : {errs[-1]}")
