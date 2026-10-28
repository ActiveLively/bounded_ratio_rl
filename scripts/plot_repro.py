#!/usr/bin/env python
"""
Plots of the BPO reproduction next to the paper's figures (arXiv:2604.18578v3).

Reads the local TensorBoard event files written by the runs that scripts/repro_launcher.py
launched (SB3: <out>/sb3/runs/<wandb id>/, IsaacLab: logs/rsl_rl/<task>/<date>_<name>/) and
redraws Fig. 4, 5, 7, 8, 9 and Table 1 with the same panels, axes and horizons.

    python scripts/plot_repro.py --out ~/bpo_repro [--paper paper.pdf]

With --paper, each figure also gets a "paper vs ours" image with the paper's figure on top.
"""
import argparse
import csv
import glob
import os
import pickle
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SB3_TAGS = ["rollout/ep_rew_mean", "train/ratio_high", "train/ratio_low"]
RSL_TAGS = ["Train/mean_reward", "Loss/learning_rate"]
# paper's colours/styles: BPO solid blue, PPO dashed orange
STYLE = {"bpo": dict(color="#1f77b4", ls="-", label="BPO"),
         "ppo": dict(color="#ffa500", ls="--", label="PPO")}
ISAAC_DIR = {"go1_rough": "unitree_go1_rough"}

# Table 1 of the paper
PAPER_T1 = {
    "Ant-v4": {"bpo": 4871.4, "ppo": 4230.1, "sac": 6161.8},
    "Humanoid-v4": {"bpo": 1570.4, "ppo": 781.3, "sac": 6806.4},
    "Hopper-v4": {"bpo": 3505.1, "ppo": 2497.7, "sac": 3015.1},
    "Swimmer-v4": {"bpo": 354.6, "ppo": 362.4, "sac": 102.7},
    "BreakoutNoFrameskip-v4": {"bpo": 374.6, "ppo": 360.4, "dqn": 252.5},
    "PongNoFrameskip-v4": {"bpo": 20.6, "ppo": 20.6, "dqn": 20.6},
    "BoxingNoFrameskip-v4": {"bpo": 94.7, "ppo": 95.7, "dqn": 92.5},
    "AsterixNoFrameskip-v4": {"bpo": 11247.9, "ppo": 9471.5, "dqn": 7122.8},
}

# page (1-based) and crop box (fractions of the page: left, top, right, bottom) of each paper figure
PAPER_CROPS = {
    "fig4": (10, (0.17, 0.08, 0.83, 0.32)),
    "fig5": (11, (0.17, 0.08, 0.83, 0.19)),
    "fig7": (11, (0.17, 0.70, 0.83, 0.83)),
    "fig8": (12, (0.42, 0.16, 0.83, 0.30)),
    "fig9": (12, (0.17, 0.455, 0.83, 0.58)),
}


def read_events(paths, tags):
    """Concatenate scalars for `tags` over event files (a resumed run has several)."""
    out = {t: [] for t in tags}
    for p in paths:
        ea = EventAccumulator(p, size_guidance={"scalars": 0})
        ea.Reload()
        have = ea.Tags()["scalars"]
        for t in tags:
            if t in have:
                out[t] += [(e.step, e.value) for e in ea.Scalars(t)]
    res = {}
    for t, v in out.items():
        if v:
            a = np.array(sorted(dict(v).items()), dtype=float)
            res[t] = (a[:, 0], a[:, 1])
    return res


def _load_one(job):
    key, paths, tags = job
    return key, read_events(paths, tags)


def load_all(out_dir, cache):
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    jobs = []
    # SB3 runs: map wandb id -> (env, algo, seed) via results/per_run.csv
    with open(os.path.join(out_dir, "results", "per_run.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["domain"] == "isaaclab":
                continue
            rid = r["wandb_url"].strip().rstrip("/").split("/")[-1]
            paths = sorted(glob.glob(os.path.join(out_dir, "sb3", "runs", rid, "*", "events*")))
            if paths:
                jobs.append(((r["env"], r["algo"], int(r["seed"])), paths, SB3_TAGS))
    # IsaacLab runs: a restarted job leaves several dirs; keep the longest one
    best = {}
    for p in glob.glob(os.path.join(REPO, "logs", "rsl_rl", "*", "*", "events*")):
        m = re.match(r"[\d_-]+_(\w+_rough)-(.+)-s(\d+)$", os.path.basename(os.path.dirname(p)))
        if not m:
            continue  # calibration runs have no name
        env = {v: k for k, v in ISAAC_DIR.items()}.get(m[1], m[1])
        key = (env, m[2], int(m[3]))
        best.setdefault(key, []).append(p)
    jobs += [((key, i), [p], RSL_TAGS) for key, paths in best.items() for i, p in enumerate(paths)]
    with ProcessPoolExecutor(32) as ex:
        loaded = dict(ex.map(_load_one, jobs, chunksize=4))
    data = {k: v for k, v in loaded.items() if isinstance(k[1], str)}
    for key, paths in best.items():
        runs = [loaded[(key, i)] for i in range(len(paths))]
        data[key] = max(runs, key=lambda d: d["Train/mean_reward"][0].max() if d else -1)
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


def seeds_curve(data, env, algo, tag, xmax=None, n=300):
    """Mean and std over seeds on a common step grid (up to the shortest seed or xmax)."""
    runs = [d[tag] for (e, a, _), d in data.items() if e == env and a == algo and tag in d]
    if not runs:
        return None
    hi = min(x[-1] for x, _ in runs)
    if xmax:
        hi = min(hi, xmax)
    lo = max(x[0] for x, _ in runs)
    grid = np.linspace(lo, hi, n)
    ys = np.stack([np.interp(grid, x, y) for x, y in runs])
    return grid, ys.mean(0), ys.std(0), len(runs)


def band(ax, c, log=False, **style):
    x, m, s, _ = c
    ax.plot(x, m, lw=1.4, **style)
    lo = m - s
    if log:
        lo = np.maximum(lo, m * 0.05)
    ax.fill_between(x, lo, m + s, color=style["color"], alpha=0.2, lw=0)


def finish(ax, title, xlabel="Steps"):
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def fig4(data, figdir):
    # (panel title, env, x horizon as plotted in the paper)
    panels = [("Swimmer", "Swimmer-v4", 2e7), ("Humanoid", "Humanoid-v4", 3.5e7),
              ("Hopper", "Hopper-v4", 2e7), ("Ant", "Ant-v4", 2e7),
              ("Pong", "PongNoFrameskip-v4", 2e7), ("Breakout", "BreakoutNoFrameskip-v4", 2e7),
              ("Boxing", "BoxingNoFrameskip-v4", 2e7), ("Asterix", "AsterixNoFrameskip-v4", 3e7)]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    for ax, (title, env, xmax) in zip(axs.flat, panels):
        for algo in ("bpo", "ppo"):
            c = seeds_curve(data, env, algo, "rollout/ep_rew_mean", xmax)
            if c:
                band(ax, c, **{**STYLE[algo], "label": f"{STYLE[algo]['label']} (n={c[3]})"})
        finish(ax, title)
        ax.legend(fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel("Episode reward")
    fig.suptitle("Fig. 4 (ours): BPO vs PPO on MuJoCo and Atari, mean ± std over seeds", y=1.0)
    fig.tight_layout()
    return save(fig, figdir, "fig4")


def fig5(data, figdir):
    panels = [("GO1-rough", "go1_rough", 1500), ("Anymal-C-rough", "anymal_c_rough", 1500),
              ("G1-rough", "g1_rough", 3000), ("H1-rough", "h1_rough", 3000)]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.3))
    for ax, (title, env, xmax) in zip(axs, panels):
        for algo in ("bpo", "ppo"):
            c = seeds_curve(data, env, algo, "Train/mean_reward", xmax)
            if c:
                band(ax, c, **{**STYLE[algo], "label": f"{STYLE[algo]['label']} (n={c[3]})"})
        finish(ax, title, "Iterations")
        ax.legend(fontsize=8)
    axs[0].set_ylabel("Episode reward")
    fig.suptitle("Fig. 5 (ours): BPO vs PPO on IsaacLab, mean ± std over seeds", y=1.02)
    fig.tight_layout()
    return save(fig, figdir, "fig5")


def fig7(data, figdir):
    # (title, env, x horizon, PPO clip, BPO clip as used in our sweeps, paper's y-range)
    panels = [("Swimmer: ratios", "Swimmer-v4", 4e7, 0.1, 0.1, (0.55, 1.3)),
              ("Hopper: ratios", "Hopper-v4", 3e7, 0.2, 0.25, (0.55, 1.4)),
              ("Pong: ratios", "PongNoFrameskip-v4", 1.5e7, 0.2, 0.3, (0.4, 1.35)),
              ("Asterix: ratios", "AsterixNoFrameskip-v4", 4e7, 0.2, 0.3, (0.35, 1.45))]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.3))
    for ax, (title, env, xmax, clip, bpo_clip, ylim) in zip(axs, panels):
        for algo in ("bpo", "ppo"):
            for side, alpha in (("high", 1.0), ("low", 0.6)):
                c = seeds_curve(data, env, algo, f"train/ratio_{side}", xmax)
                if c:
                    st = {**STYLE[algo], "label": f"{STYLE[algo]['label']}_{side}"}
                    x, m, s, _ = c
                    ax.plot(x, m, lw=1.3, alpha=alpha, color=st["color"], ls=st["ls"], label=st["label"])
                    ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.15, lw=0)
        for y in (1.0, 1 - clip, 1 + clip):
            ax.axhline(y, color="k", ls="--", lw=0.8)
        for y in (1 - bpo_clip, 1 + bpo_clip):
            ax.axhline(y, color="gray", ls="--", lw=0.8)
        ax.set_ylim(*ylim)
        finish(ax, title)
        ax.legend(fontsize=7, ncol=2, loc="lower left")
    axs[0].set_ylabel("Ratio")
    fig.suptitle("Fig. 7 (ours): mean ratio π/π₀ above / below 1 after each update, mean ± std over seeds "
                 "(black dashed: 1 and PPO clip range; gray: BPO clip range)", y=1.02)
    fig.tight_layout()
    return save(fig, figdir, "fig7")


def fig8(data, figdir):
    panels = [("Anymal-C: adapted lr", "anymal_c_rough", 1500), ("G1-rough: adapted lr", "g1_rough", 3000)]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, (title, env, xmax) in zip(axs, panels):
        for algo in ("bpo", "ppo"):
            c = seeds_curve(data, env, algo, "Loss/learning_rate", xmax)
            if c:
                band(ax, c, log=True, **STYLE[algo])
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 2e-2)
        finish(ax, title, "Iterations")
        ax.legend(fontsize=8)
    axs[0].set_ylabel("Learning rate")
    fig.suptitle("Fig. 8 (ours): KL-adaptive learning rate in RSL-RL", y=1.02)
    fig.tight_layout()
    return save(fig, figdir, "fig8")


def fig9(data, figdir):
    # colours follow the paper's panels
    navy, green, blue, light, purple = "#1f3a68", "#2ca02c", "#2b6cb0", "#7fb8e6", "#9467bd"
    panels = [
        ("ablation: mean vs median value", [("bpo", "BPO", "#1f3a68", "-"), ("ppo", "PPO", "#ffa500", "--"),
                                            ("bpo_value", "BPO_value", "#3a8fd6", "-")]),
        ("ablation: loss", [("bpo", "BPO", "#1f3a68", "-"), ("ppo", "PPO", "#ffa500", "--"),
                            ("bpo_tv", "BPO_TV", "#3a8fd6", "-"), ("bpo_rkl", "BPO_RKL", "#8cc2ea", "-")]),
        ("ablation: lambda", [("bpo", "0.001", navy, "-"), ("bpo_lam0.01", "0.01", green, "-"),
                              ("bpo_lam0.03", "0.03", blue, "-"), ("bpo_lam0.1", "0.1", light, "-")]),
        ("ablation: TV loss coef", [("bpo_tvcoef0.1", "0.1", navy, "-"), ("bpo_tvcoef0.3", "0.3", green, "-"),
                                    ("bpo_tvcoef1.0", "1.0", blue, "-"), ("bpo_tvcoef3.0", "3.0", light, "-"),
                                    ("bpo_tvcoef10.0", "10.0", purple, "-")]),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.3))
    for ax, (title, lines) in zip(axs, panels):
        for algo, label, color, ls in lines:
            c = seeds_curve(data, "g1_rough", algo, "Train/mean_reward", 3000)
            if c:
                band(ax, c, color=color, ls=ls, label=f"{label} (n={c[3]})")
        finish(ax, title, "Iterations")
        ax.legend(fontsize=7)
        # paper's y-range; a single-iteration reward spike in one λ=0.1 seed would otherwise flatten the panel
        lo = min(min(y) for y in (l.get_ydata() for l in ax.get_lines()))
        if lo < -12:
            ax.set_ylim(-12, 42)
            ax.text(0.98, 0.02, f"clipped: mean dips to {lo:.0f} (one seed, one iteration)", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6, color="gray")
    axs[0].set_ylabel("Episode reward")
    fig.suptitle("Fig. 9 (ours): G1-rough ablations, mean ± std over seeds", y=1.02)
    fig.tight_layout()
    return save(fig, figdir, "fig9")


def table1(out_dir, figdir):
    """Grouped bars: paper value (hatched) vs ours (solid, mean ± std over seeds) per env and algo."""
    rows = {}
    with open(os.path.join(out_dir, "results", "per_run.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["env"] in PAPER_T1 and r["algo"] in ("bpo", "ppo", "sac", "dqn"):
                rows.setdefault((r["env"], r["algo"]), []).append(float(r["final_reward"]))
    colors = {"bpo": "#1f77b4", "ppo": "#ffa500", "sac": "#2ca02c", "dqn": "#9467bd"}
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.5))
    md = ["| Env | Algo | Paper | Ours (mean ± std) | Ours / paper |", "|---|---|---|---|---|"]
    for ax, (env, paper) in zip(axs.flat, PAPER_T1.items()):
        algos = list(paper)
        x = np.arange(len(algos))
        ours = [np.array(rows.get((env, a), [np.nan])) for a in algos]
        ax.bar(x - 0.2, [paper[a] for a in algos], 0.38, color=[colors[a] for a in algos],
               alpha=0.35, hatch="//", edgecolor="k", lw=0.5, label="paper")
        ax.bar(x + 0.2, [o.mean() for o in ours], 0.38, yerr=[o.std(ddof=1) for o in ours], capsize=3,
               color=[colors[a] for a in algos], edgecolor="k", lw=0.5, label="ours")
        ax.set_xticks(x, [a.upper() for a in algos])
        ax.set_title(env.replace("NoFrameskip", ""), fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for a, o in zip(algos, ours):
            md.append(f"| {env} | {a.upper()} | {paper[a]:g} | {o.mean():.1f} ± {o.std(ddof=1):.1f} | "
                      f"{o.mean() / paper[a]:.2f} |")
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].set_ylabel("Final episode reward")
    axs[1, 0].set_ylabel("Final episode reward")
    fig.suptitle("Table 1: paper (hatched) vs ours (solid, mean ± std over seeds). "
                 "Ours: SAC/DQN stopped at the 48 h cap; Asterix compared at 40M steps, not 12 h",
                 y=1.0, fontsize=10)
    fig.tight_layout()
    with open(os.path.join(figdir, "table1.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    return save(fig, figdir, "table1")


def save(fig, figdir, name):
    path = os.path.join(figdir, f"{name}_ours.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def paper_vs_ours(paper_pdf, figdir, name, ours_png):
    page, (l, t, r, b) = PAPER_CROPS[name]
    stem = os.path.join(figdir, f".page{page}")
    if not os.path.exists(stem + ".png"):
        subprocess.run(["pdftoppm", "-r", "220", "-f", str(page), "-l", str(page), "-png",
                        "-singlefile", paper_pdf, stem], check=True)
    pg = plt.imread(stem + ".png")
    h, w = pg.shape[:2]
    crop = pg[int(t * h):int(b * h), int(l * w):int(r * w)]
    ours = plt.imread(ours_png)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 14 * (crop.shape[0] / crop.shape[1]
                                                          + ours.shape[0] / ours.shape[1]) + 0.8),
                                 gridspec_kw={"height_ratios": [crop.shape[0] / crop.shape[1],
                                                                ours.shape[0] / ours.shape[1]]})
    for ax, img, title in ((a1, crop, "Paper"), (a2, ours, "Ours")):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(figdir, f"{name}_paper_vs_ours.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/bpo_repro"))
    ap.add_argument("--paper", help="paper PDF, for the side-by-side images")
    ap.add_argument("--refresh", action="store_true", help="re-read event files instead of the cache")
    args = ap.parse_args()
    figdir = os.path.join(args.out, "results", "figures")
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(args.out, "results", "curves.pkl")
    if args.refresh and os.path.exists(cache):
        os.remove(cache)
    data = load_all(args.out, cache)
    print(f"loaded {len(data)} runs")
    figs = {"fig4": fig4(data, figdir), "fig5": fig5(data, figdir), "fig7": fig7(data, figdir),
            "fig8": fig8(data, figdir), "fig9": fig9(data, figdir)}
    print(table1(args.out, figdir))
    for name, png in figs.items():
        print(png)
        if args.paper:
            print(paper_vs_ours(args.paper, figdir, name, png))


if __name__ == "__main__":
    main()
