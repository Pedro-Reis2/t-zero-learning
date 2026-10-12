"""Pull runs from wandb (project ppo-assignment) and build the report figures + summary.

Usage:  python scripts/plot_wandb.py
Output: report/figs/*.png and report/summary.md
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()
PROJECT = os.environ.get("WANDB_PROJECT", "ppo-assignment")
OUT = Path("report/figs")
OUT.mkdir(parents=True, exist_ok=True)

KEYS = [
    "global_step", "_runtime", "charts/episodic_return_mean_last100", "losses/approx_kl", "losses/clipfrac",
    "losses/entropy", "losses/explained_variance", "losses/value_loss", "losses/policy_loss",
    "charts/advantage_std", "charts/SPS", "charts/num_updates", "losses/grad_norm",
]

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY") or api.default_entity
runs = [r for r in api.runs(f"{entity}/{PROJECT}") if r.state == "finished"]
relevant = {"ppo_inicial", "ppo_2M", "ppo_2M_gamma", "ppo_final", "a2c_lunarlander"}

data = {}  # (exp_name, seed) -> (df, summary)
for r in runs:
    exp, seed = r.config.get("exp_name"), r.config.get("seed")
    if exp not in relevant:
        continue
    print(f"fetching {exp} seed={seed}...", flush=True)
    df = r.history(samples=5000)
    if "global_step" not in df.columns and "_step" in df.columns:
        df["global_step"] = df["_step"]
    df = df[[k for k in KEYS if k in df.columns]].dropna(subset=["global_step"]).sort_values("global_step")
    key = (exp, seed)
    if key in data and data[key][2] > r.created_at:  # keep the most recent run per (exp, seed)
        continue
    data[key] = (df, dict(r.summary), r.created_at)
    print("loaded", exp, seed, len(df), flush=True)

COLORS = {}
PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def color(exp):
    if exp not in COLORS:
        COLORS[exp] = PALETTE[len(COLORS) % len(PALETTE)]
    return COLORS[exp]


def plot(exps, metric, fname, title=None, x="global_step", smooth=1, logy=False):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for exp in exps:
        for (e, seed), (df, _, _) in sorted(data.items(), key=lambda kv: str(kv[0])):
            if e != exp or metric not in df.columns:
                continue
            d = df[[x, metric]].dropna()
            y = d[metric].rolling(smooth, min_periods=1).mean() if smooth > 1 else d[metric]
            xs = d[x] / 60.0 if x == "_runtime" else d[x]
            ax.plot(xs, y, color=color(exp), ls="-" if seed == 1 else "--", lw=1.1, label=f"{exp} s{seed}")
    ax.set_xlabel("tempo (min)" if x == "_runtime" else "passos de ambiente")
    ax.set_ylabel(metric.split("/")[-1])
    if logy:
        ax.set_yscale("log")
    if "return" in metric:
        ax.axhline(200, color="gray", lw=0.7, ls=":")
    ax.set_title(title or metric, fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def first_step(df, thr):
    d = df[["global_step", "charts/episodic_return_mean_last100"]].dropna()
    hit = d[d["charts/episodic_return_mean_last100"] >= thr]
    return int(hit["global_step"].iloc[0]) if len(hit) else None


# ---------------- summary table ----------------
lines = ["| exp | seed | ret_last100 final | eval/mean_return | passos p/ 0 | p/ 100 | p/ 200 | tempo (min) | SPS | num_updates |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for (exp, seed), (df, s, _) in sorted(data.items(), key=lambda kv: str(kv[0])):
    ret = df["charts/episodic_return_mean_last100"].dropna()
    f = lambda v: "—" if v is None else f"{v/1e6:.2f}M"
    lines.append(
        f"| {exp} | {seed} | {ret.iloc[-1]:.1f} | {s.get('eval/mean_return', float('nan')):.1f} ± {s.get('eval/std_return', float('nan')):.1f} "
        f"| {f(first_step(df, 0))} | {f(first_step(df, 100))} | {f(first_step(df, 200))} "
        f"| {s.get('_runtime', 0)/60:.1f} | {df['charts/SPS'].dropna().iloc[-1]:.0f} | {int(df['charts/num_updates'].dropna().iloc[-1]) if 'charts/num_updates' in df else '—'} |"
    )
Path("report/summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# ---------------- figures ----------------
exps = sorted({e for e, _ in data})
ppo_path = [e for e in ["ppo_inicial", "ppo_2M", "ppo_2M_gamma", "ppo_final"] if e in exps]
R = "charts/episodic_return_mean_last100"
plot(["ppo_inicial", "ppo_final"], R, "p4_return_inicial_vs_final.png", "Retorno: config inicial vs final (2 seeds)")
plot(ppo_path, R, "p4_return_path.png", "Retorno ao longo das mudanças")
for m, fn in [("losses/approx_kl", "p4_approx_kl.png"), ("losses/clipfrac", "p4_clipfrac.png"),
              ("losses/explained_variance", "p4_explained_variance.png"), ("charts/advantage_std", "p4_advantage_std.png"),
              ("losses/entropy", "p4_entropy.png"), ("losses/value_loss", "p4_value_loss.png")]:
    plot(ppo_path, m, fn, smooth=5, logy=m in ("losses/value_loss", "losses/approx_kl"))
cmp = [e for e in ["ppo_final", "a2c_lunarlander"] if e in exps]
plot(cmp, R, "p5_return_steps.png", "PPO vs A2C: retorno × passos")
plot(cmp, R, "p5_return_time.png", "PPO vs A2C: retorno × tempo", x="_runtime")
plot(cmp, "charts/SPS", "p5_sps.png", "PPO vs A2C: SPS")
plot(cmp, "charts/num_updates", "p5_num_updates.png", "PPO vs A2C: passos de gradiente", logy=True)
plot(cmp, "losses/explained_variance", "p5_explained_variance.png", smooth=5)
plot(cmp, "losses/entropy", "p5_entropy.png", smooth=5)
print("figures in", OUT)
