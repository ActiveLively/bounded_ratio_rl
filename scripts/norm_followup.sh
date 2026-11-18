#!/usr/bin/env bash
# Does the Humanoid normalisation result (bpo_norm 4255 vs ppo_norm 2109) carry over to other tasks?
# Hopper-v4 and Ant-v4, bpo_norm / ppo_norm = reproduction settings + VecNormalize, 5 seeds, 5M steps, CPU.
# norm_gamma = each sweep's algo.gamma. Finished runs are skipped.
# Compare with: python scripts/plot_abpo.py --variants bpo_norm ppo_norm
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs_abpo/_logs
STEPS=${STEPS:-5000000}
echo "=== norm follow-up start $(date -Is)"
for seed in 1 2 3 4 5; do
  for spec in "Hopper-v4 Hopper-V4 bpo 0.999" "Hopper-v4 Hopper-V4 ppo 0.999" "Ant-v4 Ant-V4 bpo 0.99" "Ant-v4 Ant-V4 ppo 0.98"; do
    set -- $spec
    out="runs_abpo/$1/$3_norm/s$seed"
    if [ -f "$out/DONE" ]; then continue; fi
    ( python scripts/train_local.py --out "$out" \
        --sweep "sweeps/mujoco/$2/$3_seed.yaml" training.seed=$seed training.total_timesteps=$STEPS \
        env.normalize=true env.norm_gamma=$4 > "runs_abpo/_logs/$1-$3_norm-s$seed.log" 2>&1 && touch "$out/DONE" ) &
  done
done
wait
echo "NORM_DONE $(date -Is)"
