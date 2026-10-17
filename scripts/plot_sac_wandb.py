"""Pull finished runs from wandb (project sac-assignment), print exact tables, and build figures.

Usage:
    python scripts/plot_sac_wandb.py [best_exp_name]
Outputs:
    report_sac/summary.md
    report_sac/figs/*.png
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

PROJECT = "sac-assignment"
OUT = Path("report_sac/figs")
OUT.mkdir(parents=True, exist_ok=True)

KEYS = [
    "global_step", "_runtime",
    "charts/episodic_return_mean_last100",
    "charts/episodic_length_mean_last100",
    "charts/SPS", "charts/num_updates",
    "losses/alpha", "losses/alpha_loss", "losses/entropy", "charts/target_entropy",
    "losses/qf1_values", "losses/qf2_values", "losses/qf1_loss", "losses/qf2_loss", "losses/actor_loss",
    "charts/q_estimate_ep_start", "charts/soft_return_ep_start", "charts/q_minus_return",
]

api = wandb.Api(timeout=60)
entity = os.environ.get("WANDB_ENTITY") or api.default_entity
print(f"Querying {entity}/{PROJECT}...", flush=True)
runs = [r for r in api.runs(f"{entity}/{PROJECT}") if r.state == "finished"]

data = {}  # (exp_name, seed) -> dict(df, summary, config, name, id, created_at)
for r in runs:
    exp = r.config.get("exp_name")
    algo = r.config.get("algorithm", "")
    if (not exp or exp == "empty") and algo == "ppo_continuous_action":
        exp = "ppo_hopper"
    seed = int(r.config.get("seed", 0))
    if not exp:
        continue
    key = (exp, seed)
    if key in data and data[key]["created_at"] > r.created_at:
        continue
    print(f"Fetching {exp} s{seed} ({r.name}, id={r.id})...", flush=True)
    df = r.history(samples=2500)
    if "global_step" not in df.columns and "_step" in df.columns:
        df["global_step"] = df["_step"]
    keep_cols = [k for k in KEYS if k in df.columns]
    df = df[keep_cols].dropna(subset=["global_step"]).sort_values("global_step")
    data[key] = {
        "df": df,
        "summary": dict(r.summary),
        "config": dict(r.config),
        "name": r.name,
        "id": r.id,
        "created_at": r.created_at,
    }

# Build per-run and per-config summary
lines = [
    "# SAC Assignment — Exact Run Summary\n",
    "## Individual Runs",
    "| exp_name | seed | wandb run name | id | eval/mean_return | eval/mean_return_sampled | last100 final | alpha final | entropy final | q_minus_ret mean | q_minus_ret final | qf1 final | SPS | runtime (min) | num_updates |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]

by_exp = {}
for (exp, seed), info in sorted(data.items()):
    df = info["df"]
    s = info["summary"]
    ev = s.get("eval/mean_return", float("nan"))
    ev_std = s.get("eval/std_return", float("nan"))
    ev_samp = s.get("eval/mean_return_sampled", float("nan"))
    r100 = df["charts/episodic_return_mean_last100"].dropna().iloc[-1] if "charts/episodic_return_mean_last100" in df and len(df["charts/episodic_return_mean_last100"].dropna()) else float("nan")
    alpha_f = df["losses/alpha"].dropna().iloc[-1] if "losses/alpha" in df and len(df["losses/alpha"].dropna()) else float("nan")
    ent_f = df["losses/entropy"].dropna().iloc[-1] if "losses/entropy" in df and len(df["losses/entropy"].dropna()) else float("nan")
    qmr = df["charts/q_minus_return"].dropna().mean() if "charts/q_minus_return" in df and len(df["charts/q_minus_return"].dropna()) else float("nan")
    qmr_f = df["charts/q_minus_return"].dropna().iloc[-1] if "charts/q_minus_return" in df and len(df["charts/q_minus_return"].dropna()) else float("nan")
    qf1 = df["losses/qf1_values"].dropna().iloc[-1] if "losses/qf1_values" in df and len(df["losses/qf1_values"].dropna()) else float("nan")
    sps = df["charts/SPS"].dropna().iloc[-1] if "charts/SPS" in df and len(df["charts/SPS"].dropna()) else float("nan")
    rt = s.get("_runtime", df["_runtime"].dropna().iloc[-1] if "_runtime" in df else 0) / 60.0
    nu = int(df["charts/num_updates"].dropna().iloc[-1]) if "charts/num_updates" in df and len(df["charts/num_updates"].dropna()) else 0
    lines.append(
        f"| {exp} | {seed} | `{info['name']}` | `{info['id']}` | {ev:.1f} ± {ev_std:.1f} | {ev_samp:.1f} | {r100:.1f} | {alpha_f:.4f} | {ent_f:.2f} | {qmr:.1f} | {qmr_f:.1f} | {qf1:.1f} | {sps:.0f} | {rt:.1f} | {nu} |"
    )
    by_exp.setdefault(exp, []).append((seed, ev, r100, alpha_f, qmr, qf1, sps, rt, nu, info))

lines.append("\n## Aggregated by Config (Mean and [Min, Max] across seeds)")
lines.append("| exp_name | seeds | eval/mean_return: mean (min–max) | r_last100: mean (min–max) | alpha final mean | q_minus_ret mean | qf1 final mean | runtime mean (min) |")
lines.append("|---|---|---|---|---|---|---|---|")
for exp, items in sorted(by_exp.items()):
    evs = [x[1] for x in items]
    r100s = [x[2] for x in items]
    als = [x[3] for x in items]
    qmrs = [x[4] for x in items]
    qf1s = [x[5] for x in items]
    rts = [x[7] for x in items]
    lines.append(
        f"| {exp} | {len(items)} | **{np.mean(evs):.1f}** ({np.min(evs):.1f}–{np.max(evs):.1f}) "
        f"| **{np.mean(r100s):.1f}** ({np.min(r100s):.1f}–{np.max(r100s):.1f}) "
        f"| {np.nanmean(als):.4f} | {np.nanmean(qmrs):.1f} | {np.nanmean(qf1s):.1f} | {np.mean(rts):.1f} |"
    )

summary_text = "\n".join(lines)
Path("report_sac/summary.md").write_text(summary_text, encoding="utf-8")
print(summary_text, flush=True)

best_exp = sys.argv[1] if len(sys.argv) > 1 else "sac_hopper"
print(f"\nSelected Part 3 final config for plots: {best_exp}", flush=True)

PALETTE = {
    "sac_hopper": "#1f77b4",
    "sac_ls1000": "#ff7f0e",
    "sac_tau001": "#2ca02c",
    "sac_pf1": "#9467bd",
    "sac_utd2": "#2ca02c",
    "sac_no_autotune": "#d62728",
    "sac_single_q": "#d62728",
    "ppo_hopper": "#ff7f0e",
}
LS = {1: "-", 2: "--", 3: ":"}


def plot_series(ax, exps, metric, x="global_step", smooth=5, ylabel=None, title=None, logy=False, seeds_filter=None):
    for exp in exps:
        if exp not in by_exp:
            continue
        col = PALETTE.get(exp, "#333333")
        for item in sorted(by_exp[exp], key=lambda z: z[0]):
            seed, info = item[0], item[9]
            if seeds_filter is not None and seed not in seeds_filter:
                continue
            df = info["df"]
            if metric not in df.columns or x not in df.columns:
                continue
            d = df[[x, metric]].dropna()
            if d.empty:
                continue
            xs = d[x] / 60.0 if x == "_runtime" else d[x] / 1e3
            ys = d[metric].rolling(smooth, min_periods=1).mean() if smooth > 1 else d[metric]
            lbl = f"{exp} s{seed} ({info['id']})"
            ax.plot(xs, ys, color=col, ls=LS.get(seed, "-"), lw=1.3, alpha=0.9, label=lbl)
    ax.set_xlabel("Tempo de relógio (min)" if x == "_runtime" else "Passos de ambiente (×10³)", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, fontsize=8.5, fontweight="bold")
    if logy:
        ax.set_yscale("log")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5.5, loc="best")


# 1. Part 3 figure: sac_hopper (initial/final) vs closest candidates (sac_pf1 & sac_utd2), 3 seeds each
if "sac_hopper" in by_exp:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1))
    exps_p3 = [e for e in ["sac_hopper", "sac_pf1", "sac_utd2"] if e in by_exp]
    plot_series(
        axes[0], exps_p3, "charts/episodic_return_mean_last100",
        smooth=5, ylabel="Retorno médio (últimos 100 ep.)",
        title="(a) Curvas individuais por seed (3 seeds por config)",
    )
    ax = axes[1]
    for exp in exps_p3:
        col = PALETTE.get(exp, "#333333")
        grid = np.linspace(5, 100, 100)
        curves = []
        for item in by_exp[exp]:
            df = item[9]["df"]
            d = df[["global_step", "charts/episodic_return_mean_last100"]].dropna()
            if not d.empty:
                curves.append(np.interp(grid * 1e3, d["global_step"], d["charts/episodic_return_mean_last100"]))
        if curves:
            arr = np.vstack(curves)
            ax.plot(grid, arr.mean(axis=0), color=col, lw=2.0, label=f"{exp} (média 3 seeds)")
            ax.fill_between(grid, arr.min(axis=0), arr.max(axis=0), color=col, alpha=0.18, label=f"{exp} (faixa mín–máx)")
    ax.set_xlabel("Passos de ambiente (×10³)", fontsize=8)
    ax.set_ylabel("Retorno médio (últimos 100 ep.)", fontsize=8)
    ax.set_title("(b) Média e faixa mín–máx (sobreposição entre seeds)", fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "p3_return_inicial_vs_final.png", dpi=160)
    plt.close(fig)

# 2. Part 4 Ablation 1 figure: autotune vs sac_no_autotune (seeds 1,2)
if "sac_no_autotune" in by_exp:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.7))
    pair = [best_exp, "sac_no_autotune"]
    plot_series(axes[0], pair, "losses/alpha", smooth=3, ylabel="Temperatura α", title="(a) Temperatura (losses/alpha)", seeds_filter={1, 2})
    plot_series(axes[1], pair, "losses/entropy", smooth=5, ylabel="Entropia (-E[log π])", title="(b) Entropia vs target_entropy (-3.0)", seeds_filter={1, 2})
    axes[1].axhline(-3.0, color="black", ls=":", lw=1.2, label="target_entropy = -3.0")
    axes[1].legend(fontsize=5.5, loc="best")
    plot_series(axes[2], pair, "charts/episodic_return_mean_last100", smooth=5, ylabel="Retorno (last100)", title="(c) Retorno (episodic_return_mean_last100)", seeds_filter={1, 2})
    fig.tight_layout()
    fig.savefig(OUT / "p4_ablation_autotune.png", dpi=160)
    plt.close(fig)

# 3. Part 4 Ablation 2 figure: clipped_double_q vs sac_single_q (seeds 1,2)
if "sac_single_q" in by_exp:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.7))
    pair = [best_exp, "sac_single_q"]
    plot_series(axes[0], pair, "charts/q_minus_return", smooth=5, ylabel="Q(s0,a0) - G_soft", title="(a) Viés Q - Retorno (q_minus_return)", seeds_filter={1, 2})
    axes[0].axhline(0.0, color="black", ls=":", lw=1.2, label="Sem viés (0)")
    axes[0].legend(fontsize=5.5, loc="best")
    plot_series(axes[1], pair, "losses/qf1_values", smooth=5, ylabel="Q1 médio no batch", title="(b) Valor Q1 (losses/qf1_values)", seeds_filter={1, 2})
    plot_series(axes[2], pair, "charts/episodic_return_mean_last100", smooth=5, ylabel="Retorno (last100)", title="(c) Retorno (episodic_return_mean_last100)", seeds_filter={1, 2})
    fig.tight_layout()
    fig.savefig(OUT / "p4_ablation_single_q.png", dpi=160)
    plt.close(fig)

# 4. Part 5 figure: SAC (utd=1 and utd=2) vs PPO continuous on Hopper-v5 (seeds 1,2)
if "ppo_hopper" in by_exp and best_exp in by_exp:
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.7))
    pair_step = [best_exp, "ppo_hopper"]
    pair_all = [best_exp, "sac_utd2", "ppo_hopper"] if "sac_utd2" in by_exp else pair_step
    plot_series(axes[0], pair_step, "charts/episodic_return_mean_last100", x="global_step", smooth=3, ylabel="Retorno (last100)", title="(a) Eficiência amostral (× Passos)", seeds_filter={1, 2})
    plot_series(axes[1], pair_all, "charts/episodic_return_mean_last100", x="_runtime", smooth=3, ylabel="Retorno (last100)", title="(b) Tempo de relógio (× Minutos)", seeds_filter={1, 2})
    plot_series(axes[2], pair_all, "charts/SPS", x="global_step", smooth=3, ylabel="Passos/s (SPS)", title="(c) Throughput (charts/SPS)", logy=True, seeds_filter={1, 2})
    plot_series(axes[3], pair_all, "charts/num_updates", x="global_step", smooth=1, ylabel="Gradientes acumulados", title="(d) Passos de gradiente (num_updates)", logy=True, seeds_filter={1, 2})
    fig.tight_layout()
    fig.savefig(OUT / "p5_sac_vs_ppo.png", dpi=160)
    plt.close(fig)

print("Plots saved to", OUT, flush=True)

