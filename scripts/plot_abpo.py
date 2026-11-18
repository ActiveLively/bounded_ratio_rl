"""
Compare new BPO variants with the reproduced PPO and BPO baselines.

Baselines come from repro/results/curves.npz (lab-server reproduction). New runs are read from
  runs_abpo/<env>/<variant>/s<seed>/progress.csv          (SB3, written by scripts/train_local.py)
  logs/rsl_rl/<experiment>/<time>_<env>-<variant>-s<seed>  (IsaacLab, TensorBoard)

    python scripts/plot_abpo.py                 # all envs that have new runs
    python scripts/plot_abpo.py --variants abpo # only these variants

Writes abpo_results/comparison.png, one panel per env (PPO, BPO, new variants; mean +- std over
seeds), diagnostics.png (ratio bound, learning rate, KL of the new runs) and summary.md.
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISAAC_EXPERIMENT = {"unitree_go1_rough": "go1_rough"}
ENV_ORDER = ["Swimmer-v4", "Humanoid-v4", "Hopper-v4", "Ant-v4",
             "PongNoFrameskip-v4", "BreakoutNoFrameskip-v4", "BoxingNoFrameskip-v4", "AsterixNoFrameskip-v4",
             "go1_rough", "anymal_c_rough", "g1_rough", "h1_rough"]
BASE_STYLE = {"ppo": dict(color="#ffa500", ls="--", label="PPO"),
              "bpo": dict(color="#1f77b4", ls="-", label="BPO")}
NEW_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
SB3_TAGS = {"rollout/ep_rew_mean": "reward", "train/clip_ratio": "clip", "train/approx_kl": "kl",
            "train/lambda": "lambda", "train/std": "std", "train/learning_rate": "lr"}
RSL_TAGS = {"Train/mean_reward": "reward", "Loss/clip_param": "clip", "Loss/kl_iter_mean": "kl",
            "Loss/reg_w": "lambda", "Policy/mean_noise_std": "std", "Loss/learning_rate": "lr",
            "Curriculum/terrain_levels": "terrain"}


def load_baselines():
    z = np.load(os.path.join(REPO, "repro", "results", "curves.npz"))
    data = {}
    for name, env, algo, seed, tag in json.loads(str(z["index"])):
        if tag in ("rollout/ep_rew_mean", "Train/mean_reward"):
            data.setdefault((env, algo), {})[seed] = {"reward": (z[name + "_x"], z[name + "_y"].astype(float))}
        elif tag == "Loss/learning_rate":
            data.setdefault((env, algo), {}).setdefault(seed, {})["lr"] = (z[name + "_x"], z[name + "_y"].astype(float))
    return data


def load_sb3(root):
    data = {}
    for path in glob.glob(os.path.join(root, "*", "*", "s*", "progress.csv")):
        env, variant, seed = path.split(os.sep)[-4:-1]
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        run = {}
        for tag, short in SB3_TAGS.items():
            pts = [(float(r["time/total_timesteps"]), float(r[tag])) for r in rows if r.get(tag) not in (None, "")]
            if pts:
                a = np.array(pts)
                run[short] = (a[:, 0], a[:, 1])
        if "reward" in run:
            data.setdefault((env, variant), {})[int(seed[1:])] = run
    return data


def load_isaac(root):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    data = {}
    for run_dir in glob.glob(os.path.join(root, "*", "*")):
        m = re.match(r"[\d_-]+_(\w+_rough)-(.+)-s(\d+)$", os.path.basename(run_dir))
        if not m or m[2] in ("bpo", "ppo") or not glob.glob(os.path.join(run_dir, "events*")):
            continue
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
        have = ea.Tags()["scalars"]
        run = {}
        for tag, short in RSL_TAGS.items():
            if tag in have:
                ev = ea.Scalars(tag)
                run[short] = (np.array([e.step for e in ev], float), np.array([e.value for e in ev], float))
        if "reward" in run:
            key = (m[1], m[2])
            old = data.setdefault(key, {}).get(int(m[3]))
            if old is None or run["reward"][0][-1] > old["reward"][0][-1]:  # keep the longest attempt
                data[key][int(m[3])] = run
    return data


def seeds_curve(runs, tag="reward", xmax=None, n=300):
    series = [r[tag] for r in runs.values() if tag in r]
    if not series:
        return None
    hi = min(x[-1] for x, _ in series)
    if xmax:
        hi = min(hi, xmax)
    lo = max(x[0] for x, _ in series)
    grid = np.linspace(lo, hi, n)
    ys = np.stack([np.interp(grid, x, y) for x, y in series])
    return grid, ys.mean(0), ys.std(0), len(series)


def band(ax, curve, **style):
    x, m, s, n = curve
    style = {**style, "label": f"{style['label']} (n={n})"}
    ax.plot(x, m, lw=1.5, **style)
    ax.fill_between(x, m - s, m + s, color=style["color"], alpha=0.18, lw=0)


def final_scores(runs, xmax, frac=0.05):
    """Per-seed mean reward over the last `frac` of the common horizon."""
    out = []
    for r in runs.values():
        x, y = r["reward"]
        keep = (x <= xmax) & (x >= xmax * (1 - frac))
        if keep.any():
            out.append(float(y[keep].mean()))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sb3", default=os.path.join(REPO, "runs_abpo"))
    parser.add_argument("--isaac", default=os.path.join(REPO, "logs", "rsl_rl"))
    parser.add_argument("--out", default=os.path.join(REPO, "abpo_results"))
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()

    base = load_baselines()
    new = {**load_sb3(args.sb3), **load_isaac(args.isaac)}
    if args.variants:
        new = {k: v for k, v in new.items() if k[1] in args.variants}
    envs = [e for e in ENV_ORDER if any(k[0] == e for k in new)]
    if not envs:
        raise SystemExit("no new runs found")
    variants = sorted({k[1] for k in new})
    colors = {v: NEW_COLORS[i % len(NEW_COLORS)] for i, v in enumerate(variants)}
    os.makedirs(args.out, exist_ok=True)

    cols = min(4, len(envs))
    rows = -(-len(envs) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    summary = []
    for ax, env in zip(axs.flat, envs):
        # compare on the horizon every new variant has reached
        xmax = min(min(r["reward"][0][-1] for r in new[(env, v)].values()) for v in variants if (env, v) in new)
        for algo, style in BASE_STYLE.items():
            if (env, algo) in base:
                band(ax, seeds_curve(base[(env, algo)], xmax=xmax), **style)
                s = final_scores(base[(env, algo)], xmax)
                summary.append((env, style["label"], len(s), s.mean(), s.std(), xmax))
        for v in variants:
            if (env, v) in new:
                band(ax, seeds_curve(new[(env, v)], xmax=xmax), color=colors[v], ls="-", label=v)
                s = final_scores(new[(env, v)], xmax)
                summary.append((env, v, len(s), s.mean(), s.std(), xmax))
        isaac = env.endswith("_rough")
        ax.set_title(env, fontweight="bold", fontsize=10)
        ax.set_xlabel("Iterations" if isaac else "Steps")
        ax.set_ylabel("Episode reward")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axs.flat[len(envs):]:
        ax.axis("off")
    fig.suptitle("New BPO variants vs reproduced PPO and BPO (mean ± std over seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "comparison.png"), dpi=150)

    # Diagnostics of the new runs: ratio bound, KL, learning rate (with the BPO baseline's where saved)
    diag = [("clip", "Ratio bound ε", False), ("kl", "KL per update", True), ("lr", "Learning rate", True),
            ("std", "Action std", False)]
    fig, axs = plt.subplots(len(envs), len(diag), figsize=(4.0 * len(diag), 2.8 * len(envs)), squeeze=False)
    for row, env in zip(axs, envs):
        for ax, (tag, title, log) in zip(row, diag):
            if tag == "lr" and (env, "bpo") in base:
                c = seeds_curve(base[(env, "bpo")], tag)
                if c:
                    band(ax, c, **BASE_STYLE["bpo"])
            for v in variants:
                c = seeds_curve(new.get((env, v), {}), tag)
                if c:
                    band(ax, c, color=colors[v], ls="-", label=v)
            if log:
                ax.set_yscale("log")
            ax.set_title(f"{env}: {title}", fontsize=9)
            ax.grid(alpha=0.25)
            if ax.has_data():
                ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "diagnostics.png"), dpi=130)

    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("Final reward = mean over the last 5% of the compared horizon, then mean ± std over seeds.\n\n")
        f.write("| Env | Method | Seeds | Final reward | Horizon |\n|---|---|---|---|---|\n")
        for env, label, n, mean, std, xmax in summary:
            f.write(f"| {env} | {label} | {n} | {mean:.1f} ± {std:.1f} | {xmax:.3g} |\n")
    print(open(os.path.join(args.out, "summary.md")).read())


if __name__ == "__main__":
    main()
