"""Launch PPO/A2C LunarLander experiments in parallel (OMP_NUM_THREADS=1 each).

Usage (from repo root):  python scripts/run_ppo_experiments.py [round]
Logs go to logs/<name>.log
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROUNDS = {
    "1": [
        # name,              config,                 overrides
        ("ppo_inicial_s1",   "ppo_lunarlander",       ["exp_name=ppo_inicial", "seed=1"]),
        ("ppo_inicial_s2",   "ppo_lunarlander",       ["exp_name=ppo_inicial", "seed=2"]),
        ("ppo_2M_s1",        "ppo_lunarlander",       ["exp_name=ppo_2M", "total_timesteps=2000000", "seed=1"]),
        ("ppo_2M_gamma_s1",  "ppo_lunarlander",       ["exp_name=ppo_2M_gamma", "total_timesteps=2000000",
                                                       "ppo.gamma=0.999", "ppo.gae_lambda=0.98", "seed=1"]),
        ("ppo_final_s1",     "ppo_lunarlander_tuned", ["exp_name=ppo_final", "seed=1"]),
        ("ppo_final_s2",     "ppo_lunarlander_tuned", ["exp_name=ppo_final", "seed=2"]),
        ("a2c_ll_s1",        "a2c_lunarlander",       ["exp_name=a2c_lunarlander", "seed=1"]),
        ("a2c_ll_s2",        "a2c_lunarlander",       ["exp_name=a2c_lunarlander", "seed=2"]),
    ],
}


def main():
    rnd = sys.argv[1] if len(sys.argv) > 1 else "1"
    runs = ROUNDS[rnd]
    Path("logs").mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    procs = []
    for name, cfg, ov in runs:
        log = open(f"logs/{name}.log", "w", encoding="utf-8")
        cmd = [sys.executable, "train.py", "--config", cfg, "--override", *ov]
        procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        print("started", name, " ".join(cmd), flush=True)
    t0 = time.time()
    while procs:
        for item in list(procs):
            name, p, log = item
            if p.poll() is not None:
                log.close()
                print(f"finished {name} rc={p.returncode} after {time.time() - t0:.0f}s", flush=True)
                procs.remove(item)
        time.sleep(10)


if __name__ == "__main__":
    main()
