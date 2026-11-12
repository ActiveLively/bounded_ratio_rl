# Reproduction runs (condensed)

Condensed record of the October 2026 reproduction of *Bounded Ratio Reinforcement Learning*
(arXiv:2604.18578v3) on the lab server: 340 runs over MuJoCo, Atari and IsaacLab, plus two
speed-calibration passes. All 340 runs finished; none failed. What we changed relative to the paper and
to the repo as released, and why, is in [`../REPRODUCTION_NOTES.md`](../REPRODUCTION_NOTES.md).

The raw outputs (about 55 GB) were cut down to this folder (about 18 MB) by
[`scripts/condense_repro.py`](../scripts/condense_repro.py). The raw outputs were deleted from the
lab server on 2026-11-12, after checking that every run had finished uploading to wandb, so this
folder and wandb are now the only copies. Full curves, system metrics and
model files are on wandb in projects `BPO-repro-mujoco`, `BPO-repro-atari` and `BPO-repro-isaaclab`.
`results/per_run.csv` has each run's wandb URL.

## Layout

| Path | Contents |
|---|---|
| `results/summary.md`, `summary.csv` | Final reward per env and algorithm, mean ± std over seeds |
| `results/per_run.csv` | One row per run: final reward, env steps, terrain level (IsaacLab), final ratio bounds, wandb URL |
| `results/figures/` | Our versions of Fig. 4, 5, 7, 8, 9 and Table 1, and side-by-side images with the paper's figures |
| `results/curves.npz` | Reward and ratio curves for every run, downsampled to at most 500 points per series |
| `campaigns/repro/` | The main 340-run queue: `launcher.log`, launcher state `jobs-BPO-repro.json`, condensed job logs, Hydra configs of each SB3 job |
| `campaigns/calib/`, `campaigns/calib2/` | The two calibration passes (short runs used to measure speed and plan the main queue), same layout |
| `campaigns/isaaclab/` | Agent and env configs and the `git diff` recorded at launch by each IsaacLab run, stored once per distinct file; `index.csv` maps each run dir to its files. `hydra_overrides/` holds each IsaacLab launch's command-line overrides. |

Each condensed job log keeps the job's resolved config and startup output, its last metrics block,
every traceback, and the last lines of the log. Lines left out are marked `... [N lines elided] ...`.

## Redrawing the figures

`plot_repro.py` reads `results/curves.npz` when the full `curves.pkl` cache is not there
(the cache and the TensorBoard files it was built from were deleted with the raw outputs):

```bash
python scripts/plot_repro.py --out repro
```

This writes into `repro/results/figures/`. Curves drawn from the downsampled data look the same as the originals,
and Table 1 comes out identical.

## Known issues in the results

- `go1_rough` BPO never climbs the terrain curriculum (level 0–1.9, against 6.0 for PPO).
- Humanoid BPO and PPO are bimodal: most seeds get stuck at a reward of 700–1100.
- DQN and SAC stopped at the 48 h wall-clock cap, not at a step count (see P3b in the notes).
