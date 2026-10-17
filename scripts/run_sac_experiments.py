"""Launch SAC assignment experiments with a bounded worker pool (OMP_NUM_THREADS=1).

Usage:
    python scripts/run_sac_experiments.py stage1
    python scripts/run_sac_experiments.py stage2 <best_config_or_overrides> <fixed_alpha>
"""
import os
import subprocess
import sys
import time
from pathlib import Path

STAGE1 = [
    # (log_name, config, overrides)
    ("sac_pendulum_s1", "sac_pendulum", ["exp_name=sac_pendulum", "seed=1"]),
    # Part 3 — Baseline: configs/sac_hopper.yml (3 seeds)
    ("sac_hopper_s1", "sac_hopper", ["exp_name=sac_hopper", "seed=1"]),
    ("sac_hopper_s2", "sac_hopper", ["exp_name=sac_hopper", "seed=2"]),
    ("sac_hopper_s3", "sac_hopper", ["exp_name=sac_hopper", "seed=3"]),
    # Part 3 — Candidate 1: learning_starts 5000 -> 1000 (3 seeds)
    ("sac_ls1000_s1", "sac_hopper", ["exp_name=sac_ls1000", "sac.learning_starts=1000", "seed=1"]),
    ("sac_ls1000_s2", "sac_hopper", ["exp_name=sac_ls1000", "sac.learning_starts=1000", "seed=2"]),
    ("sac_ls1000_s3", "sac_hopper", ["exp_name=sac_ls1000", "sac.learning_starts=1000", "seed=3"]),
    # Part 3 — Candidate 2: tau 0.005 -> 0.01 (3 seeds)
    ("sac_tau001_s1", "sac_hopper", ["exp_name=sac_tau001", "sac.tau=0.01", "seed=1"]),
    ("sac_tau001_s2", "sac_hopper", ["exp_name=sac_tau001", "sac.tau=0.01", "seed=2"]),
    ("sac_tau001_s3", "sac_hopper", ["exp_name=sac_tau001", "sac.tau=0.01", "seed=3"]),
    # Part 3 — Candidate 3: policy_frequency 2 -> 1 (3 seeds)
    ("sac_pf1_s1", "sac_hopper", ["exp_name=sac_pf1", "sac.policy_frequency=1", "seed=1"]),
    ("sac_pf1_s2", "sac_hopper", ["exp_name=sac_pf1", "sac.policy_frequency=1", "seed=2"]),
    ("sac_pf1_s3", "sac_hopper", ["exp_name=sac_pf1", "sac.policy_frequency=1", "seed=3"]),
    # Part 3 — Candidate 4: updates_per_step 1 -> 2 (3 seeds)
    ("sac_utd2_s1", "sac_hopper", ["exp_name=sac_utd2", "sac.updates_per_step=2", "seed=1"]),
    ("sac_utd2_s2", "sac_hopper", ["exp_name=sac_utd2", "sac.updates_per_step=2", "seed=2"]),
    ("sac_utd2_s3", "sac_hopper", ["exp_name=sac_utd2", "sac.updates_per_step=2", "seed=3"]),
]


def run_pool(jobs, max_workers=9):
    Path("logs_sac").mkdir(exist_ok=True)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONIOENCODING="utf-8",
    )
    queue = list(jobs)
    active = []
    t0 = time.time()

    while queue or active:
        while queue and len(active) < max_workers:
            name, cfg, ov = queue.pop(0)
            log_file = open(f"logs_sac/{name}.log", "w", encoding="utf-8")
            cmd = [sys.executable, "train.py", "--config", cfg, "--override", *ov]
            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            active.append((name, p, log_file, time.time()))
            print(f"[{time.time()-t0:6.1f}s] STARTED {name}: {' '.join(cmd)}", flush=True)
            time.sleep(2.0)  # ensure distinct timestamps in run names

        for item in list(active):
            name, p, log_file, t_start = item
            if p.poll() is not None:
                log_file.close()
                dt = time.time() - t_start
                print(f"[{time.time()-t0:6.1f}s] FINISHED {name} rc={p.returncode} (took {dt/60:.1f}m)", flush=True)
                active.remove(item)

        if active:
            time.sleep(5.0)


STAGE2 = [
    # Part 4 — Ablation 1: autotune=false, fixed alpha=0.08 (chosen from Part 3 alpha trajectory ~0.075-0.08)
    ("sac_no_autotune_s1", "sac_hopper", ["exp_name=sac_no_autotune", "sac.autotune=false", "sac.alpha=0.08", "seed=1"]),
    ("sac_no_autotune_s2", "sac_hopper", ["exp_name=sac_no_autotune", "sac.autotune=false", "sac.alpha=0.08", "seed=2"]),
    # Part 4 — Ablation 2: clipped_double_q=false (single critic Q1)
    ("sac_single_q_s1", "sac_hopper", ["exp_name=sac_single_q", "sac.clipped_double_q=false", "seed=1"]),
    ("sac_single_q_s2", "sac_hopper", ["exp_name=sac_single_q", "sac.clipped_double_q=false", "seed=2"]),
]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    if mode == "stage1":
        run_pool(STAGE1, max_workers=9)
    elif mode == "stage2":
        extra_ov = sys.argv[2:]
        jobs = [(name, cfg, ov + extra_ov) for name, cfg, ov in STAGE2]
        run_pool(jobs, max_workers=4)

