#!/usr/bin/env bash
# Scale-free lambda on the four Atari games: lam_c0.1 = reproduction BPO settings with
# lambda = 0.1 * mean|Q - mu| instead of 0.001. 5 seeds, 10M steps, CPU (one thread per run).
# Queued behind scripts/all_bench_mujoco.sh (waits for MJ_ALL_DONE). Finished runs are skipped.
# Compare with: python scripts/plot_abpo.py --variants lam_c0.1
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs_abpo/_logs
STEPS=${STEPS:-10000000}
echo "waiting for the MuJoCo batch $(date -Is)"
until grep -q MJ_ALL_DONE runs_mjall.log 2>/dev/null || ! tmux has-session -t mjall 2>/dev/null; do sleep 30; done
echo "=== all-bench atari start $(date -Is)"
for seed in 1 2 3 4 5; do
  for game in Breakout Boxing Pong Asterix; do
    env="${game}NoFrameskip-v4"
    out="runs_abpo/$env/lam_c0.1/s$seed"
    if [ -f "$out/DONE" ]; then continue; fi
    ( python scripts/train_local.py --out "$out" --sweep sweeps/atari/bpo_seed.yaml env.env_id=$env \
        training.seed=$seed training.total_timesteps=$STEPS algo.lambda_mode=adaptive algo.lambda_=0.1 \
        > "runs_abpo/_logs/$env-lam_c0.1-s$seed.log" 2>&1 && touch "$out/DONE" ) &
  done
done
wait
echo "ATARI_DONE $(date -Is)"
