# Reproduction notes: paper vs. repo discrepancies

Running log of every place where our reproduction of *Bounded Ratio Reinforcement Learning*
(arXiv:2604.18578v3) departs from the paper or from this repo as released (commit `64ff1cd`),
and why. Update this file whenever a setting changes.

"Paper" means the PDF text and Tables 2–3 (Appendix A.9). "Repo" means the original sweep
files in `sweeps/` and configs in `cfgs/`.

## 1. Hyperparameter conflicts between paper and repo

| # | Setting | Paper | Repo (original) | Used | Reason | Files |
|---|---|---|---|---|---|---|
| H1 | Hopper BPO λ | 0.001 ("across all environments", Table 2 caption) | 10.0 (`bpo_seed`, `bpo_seed_value`) | **10.0** | λ = 10 is one of the values in the repo's own tuning grid (`bpo_sweep.yaml`: {0.001, 1, 10}) and both Hopper seed sweeps agree on it. λ scales advantages inside `sigmoid((Q − u)/λ)`, so it depends on the reward scale; Hopper uses γ = 0.999, so its returns are large. Treated as a simplification in the paper. A side check with the paper's value is run as well (see H1b). | `sweeps/mujoco/Hopper-V4/bpo_seed.yaml` (unchanged) |
| H1b | Hopper BPO λ, side check | 0.001 | n/a | **0.001, 5 seeds** | Tests whether the paper's stated value reproduces the Hopper result. CPU only, so it adds no wall-clock time. | `sweeps/mujoco/Hopper-V4/bpo_seed_lambda_paper.yaml` (new) |
| H2 | Swimmer BPO λ | 0.001 | 0.01 | **0.01** | Same reasoning as H1. Swimmer is insensitive anyway (BPO ≈ PPO ≈ 355 in Table 1). | `sweeps/mujoco/Swimmer-V4/bpo_seed.yaml` (unchanged) |
| H3 | IsaacLab BPO λ (`reg_w`) | 0.001 | 0.01 (default of `RslRlBPOAlgorithmCfg`; no task config overrides it). The algorithm class itself defaults to 0.001. | **0.001** | The Fig. 9 text treats 10⁻³ as the baseline ("increasing λ from 10⁻³ to 10⁻²…"). The Fig. 9 λ ablation still runs 0.01, 0.03 and 0.1. | `src/rsl_rl/rsl_rl_cfg.py` |
| H4 | Hopper PPO | Table 3: n_envs 4, n_epochs 10, clip 0.25, ent 0 | n_envs 1, n_epochs 5, clip 0.2, ent 0.00229519, vf 0.835671, max_grad_norm 0.7 (RL-Zoo Hopper values) | **Repo (RL-Zoo)** | Table 3's Hopper row is identical to the BPO row in Table 2, which looks like a copy error. Section 5.1 says PPO hyperparameters come from RL-Zoo, and the repo matches RL-Zoo. | `sweeps/mujoco/Hopper-V4/ppo_seed.yaml` (unchanged) |
| H5 | Atari PPO clip | 0.2 | 0.1 (constant) | **0.2** | Table 3's Atari row differs from the BPO row in clip, entropy and epochs, so it looks deliberate rather than copied. Low confidence: 0.1 is RL-Zoo's Atari value. | `sweeps/atari/ppo_seed.yaml` |

All other PPO and BPO values in Tables 2–3 match the repo: Atari BPO, MuJoCo BPO apart from λ, Ant, Humanoid and Swimmer PPO, and all IsaacLab PPO and BPO configs.

## 2. Not specified in the paper (repo behaviour kept)

| Setting | Repo | Note |
|---|---|---|
| MuJoCo observation/reward normalization | None (no `VecNormalize`), for both PPO and BPO | As far as I recall, RL-Zoo's MuJoCo PPO configs normalize. Kept so both algorithms are treated the same and PPO numbers stay comparable to the paper's. |
| MuJoCo PPO policy init | `ortho_init: true`, `log_std_init: 0.0`, ReLU, 256×256 | As far as I recall, RL-Zoo uses `ortho_init: false` and `log_std_init: -2`. |
| SAC run length (Table 1) | The paper only says "evaluated after convergence". `sac_seed.yaml` sets no length, so runs fall back to the `train.py` default of 40M steps. | **40M steps.** Same default the MuJoCo BPO/PPO sweeps use, and Fig. 7's 40M axis suggests the default was really used. SAC's Humanoid score of 6806 also points to long training. Cost: about 2.5–5 days per run at 100–200 steps/s, which makes SAC the longest job. |
| DQN run length (Table 1) | Not stated. `dqn_seed.yaml` sets no length, so runs fall back to 40M (env steps). | **40M target, stopped by the 48 h cap** (P3b) at about 5.3–5.5M steps (about 31 steps/s per run on one core). Continuing to 10M was considered and dropped (2026-10-22, time budget); the model saved at the cap can still be resumed with `training.resume` (C4). |
| Fig. 7 ratio statistics | Not logged in the repo | See C1. |

## 3. Run protocol changes

| # | Change | Paper / repo | Ours | Reason |
|---|---|---|---|---|
| P1 | MuJoCo run length | Fig. 4 plots 20M steps (Humanoid about 35M). Fig. 7 plots to 40M. Repo sweeps: 40M default, Humanoid BPO 50M, Humanoid PPO 20M. | **40M for every MuJoCo BPO and PPO run** | Covers the full Fig. 7 x-axis and treats PPO and BPO alike on Humanoid. Changed in `sweeps/mujoco/Humanoid-V4/{bpo,ppo}_seed.yaml`. |
| P2 | Atari run length | Fig. 4: 20M frames (Asterix about 30–40M; Table 1 compares Asterix after 12 h wall-clock) | 20M frames, Asterix 40M | Matches the plotted horizons. Asterix is compared at a fixed step count, not after 12 h wall-clock, because our hardware differs. |
| P3 | Fig. 9 ablation seeds (G1-rough) | 10 seeds | **5 seeds** | Compute budget. BPO and PPO curves reuse the 5-seed Fig. 5 runs. |
| P3b | **Wall-clock limits.** The authors' cluster scripts cap jobs at **24 h** (`shell_scripts/train_job_bpo`: 1 GPU + 4 CPUs per job) and **12 h** (`shell_scripts/ppo_array.sh`). Table 1 compares Asterix "after the same wall-clock time (12h)". So long runs were evidently stopped by job limits, not step counts. | n/a | SAC and DQN get a wall-clock cap (`training.max_hours`, graceful stop); **cap = 48 h** (decided 2026-10-21): twice the authors' limit, to offset our slower per-job hardware; it adds only about 3 h to the total schedule. Our runs get one contended CPU core each, which is slower per job than their 1 GPU + 4 CPUs, so the same hours mean fewer steps. Calibrated SAC rates (steps/s): Ant 146, Hopper 86, Humanoid 55, Swimmer 90, i.e. 4.8–12.6M steps in 24 h. Steps actually reached are in each run's wandb summary. | Run lengths aren't specified for SAC/DQN; the evidence points to time limits. |
| P3c | **Placement.** | n/a | IsaacLab on the 2 GPUs, one run per GPU. All SB3 runs (MuJoCo, SAC, Atari incl. DQN) on CPU, one thread each. | Calibration: sharing GPUs between 24 Atari runs and IsaacLab slowed G1 from 1.60 to 17.8 s/iteration; 3 IsaacLab runs per GPU ran at 5.4 s/iteration each, less total throughput than 1. Device choice doesn't change the algorithms, only floating-point nondeterminism. |
| P4 | Hardware | RL hardware not stated. LLM: 4× H100. | 2× H100 NVL (94 GB), 2× Xeon 8592+ (128 cores) | n/a |
| P5 | LLM experiments (Fig. 6) | Qwen2.5-Math-1.5B, GRPO vs GBPO | **Not run** | Out of scope for this round. |

## 4. Code changes

| # | Change | Files |
|---|---|---|
| C1 | **Ratio logging for Fig. 7.** After each update, one extra no-gradient pass over the rollout buffer computes π(a\|s)/π_old(a\|s) under the updated policy. The mean of the ratios above 1 is logged as `train/ratio_high`, the mean below 1 as `train/ratio_low`. BPO and PPO are measured identically. The paper does not say when its statistics were taken, so absolute levels may differ. PPO now uses `RatioStatsPPO` (stock SB3 PPO plus this logging), so its training is unchanged. | `src/sb3/ratio_stats.py`, `src/sb3/BPO.py`, `src/sb3/__init__.py`, `cfgs/algo/ppo/{mujoco,atari}.yaml` |
| C2 | **Video schedule.** One 300-step clip ends at each of 25%, 50%, 75% and 100% of training; previously a clip was recorded every 1M vectorized steps. The env is closed after training so the last clip is saved. Each finished clip is logged to wandb as media (key `videos`) by a small callback, because wandb's `monitor_gym` doesn't hook SB3 2.8's recorder. MuJoCo rendering runs on the CPU with `MUJOCO_GL=osmesa` (about 43 ms per frame), so SB3 jobs use no GPU at all; EGL would hold about 160 MB of GPU 0 per process. None of this affects results. | `train.py` |
| C3 | IsaacLab λ default 0.01 → 0.001 (see H3). | `src/rsl_rl/rsl_rl_cfg.py` |
| C4 | `training.resume` (wandb run id) in `train.py`: load the saved model, resume the same wandb run, train up to `training.total_timesteps`. Off-policy models first act for `learning_starts` steps without updates to refill the replay buffer, which isn't saved. Not used in the current runs. | `train.py`, `cfgs/config.yaml` |

## 4b. Repo issues handled in the launcher (`scripts/repro_launcher.py`)

All runs are built from the repo's sweep files by `scripts/repro_launcher.py`, a local queue used instead of wandb sweep agents. Where a sweep file can't be run as written, or a setting had to be chosen, the launcher does the following:

| # | Issue | Handling |
|---|---|---|
| L1 | `sweeps/mujoco/*/sac_seed.yaml` never sets `env=mujoco`, and `cfgs/config.yaml` defaults to `env: atari`, so as written they would build "Ant-v4" etc. as Atari games. | Launcher adds `env=mujoco`. The SAC algorithm config (`cfgs/algo/sac.yaml`) is used as is. |
| L2 | `sweeps/isaaclab/*/ppo_seed.yaml` calls Isaac Lab's `scripts/reinforcement_learning/rsl_rl/train.py`, which isn't in this repo. | Launched as `train_rsl_rl.py --algorithm ppo`, which loads Isaac Lab's built-in PPO config. Its values match Table 3. |
| L3 | **Advantage estimate in IsaacLab BPO (`online_adv`).** Not mentioned in the paper. `true` (the default) weights the loss by (Q − V) from the current value net; `false` uses the GAE advantages stored at rollout time. The sweeps set `false` for Go1 and H1 and leave `true` for G1 and Anymal-C. The G1 sweep's *name* says "offlineadv", but its parameters don't set it. | Followed the sweep **parameters**: Go1/H1 offline, G1/Anymal-C online. Sweep names are unreliable labels elsewhere too: the Anymal-C BPO name says "clip02" but its config and Table 2 use 0.25, and the Go1 name says "ent0005_kl02" but its config and Table 2 use 0.001 and 0.01. The G1 ablations inherit online advantages from the G1 BPO sweep. |
| L4 | BPO and PPO configs don't seed the networks (`seed` is only passed to the env factory); SAC and DQN configs do. | Launcher adds `++algo.seed=<seed>` for BPO/PPO so runs are reproducible. Results across seeds are statistically unaffected. |
| L5 | IsaacLab saves an 11.7 MB checkpoint every 50 iterations and uploads each to wandb: about 57 GB over the planned runs. | `agent.save_interval=500`. The final model is always saved. Training is unaffected. |
| L6 | IsaacLab runs log through RSL-RL's wandb writer (project `BPO-repro-isaaclab`); SB3 runs log through `train.py` (`BPO-repro-mujoco`, `BPO-repro-atari`), with wandb `group` = env-algorithm. | n/a |
| L7 | wandb shows runs as **crashed** that aren't. (a) IsaacLab: the process exits without `wandb.finish()` (Isaac Sim shutdown), so finished runs never send an exit status; the launcher records rc 0 and the data is complete. (b) Atari: 118 runs in one project hit wandb's per-project filestream rate limit (HTTP 429) from the start; wandb keeps retrying, and a run lagging too far is shown as crashed until its upload catches up. Everything is also in the local `.wandb` files under `<out>/sb3/wandb/`, which `wandb sync` can upload. | Check the launcher state file, not wandb's run state. |
| L8 | Restarting the launcher. SIGINT/SIGTERM stop all its jobs. To restart without that, SIGKILL the launcher process: jobs run in their own sessions and keep running, and the next launcher adopts every job marked running whose process is still alive (matched by its command line). An adopted job's exit code is unknown, so it counts as failed if its log has a traceback. | n/a |

## 5. Software environment

The paper and repo don't pin package versions, and the repo has no `uv.lock`. Resolving in October 2026 gave:

| Package | Version | Note |
|---|---|---|
| Python | 3.10.20 | Pinned by `pyproject.toml` |
| stable-baselines3 | 2.8.0 | `uv sync` resolves all extras together and backtracked to SB3 1.8.0 (old `gym==0.21`, which won't build). Installed with `uv pip install -e ".[sb3,isaaclab]" "stable-baselines3>=2.4"` instead. |
| gymnasium | 1.2.3 | Sweeps use the `-v4` MuJoCo envs (deprecated but available) |
| mujoco | 3.x | |
| torch / torchvision | 2.5.1 / 0.20.1 | Pinned by the `isaaclab` extra |
| isaaclab / isaacsim | 2.1.0 / 4.5.0.0 | Installed as `isaaclab[isaacsim,rsl-rl]`, not the README's `[all]`. `[all]` pulls in rl-games, which caps wandb below 0.13 (a 2022 release). The repo only uses RSL-RL. |
| rsl-rl-lib | 2.3.3 | |
| wandb | 0.28.0 | |

LLM / TTRL dependencies are not installed.

## 6. Open items

- If Hopper BPO with λ = 10 doesn't reproduce Table 1, compare it against H1b first.
- Not used in this round: `sweeps/mujoco/Humanoid-V4/bpo_seed_value.yaml` (λ = 0.01) and the HalfCheetah sweeps. HalfCheetah isn't in the paper.
