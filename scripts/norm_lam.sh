#!/usr/bin/env bash
# Normalised BPO with scale-free lambda (lambda = 0.1 * mean|Q - mu|), one setting on three tasks.
# Motivation: Hopper bpo_norm failed (695 vs 3302) because its tuned lambda=10 is ~50x the normalised
# advantage scale, so the target ratio is ~1 everywhere and the policy does not move (approx_kl = 0).
# Variant bpo_norm_lam0.1, 5 seeds, 5M steps, CPU. Finished runs are skipped.
# Compare with: python scripts/plot_abpo.py --variants bpo_norm ppo_norm bpo_norm_lam0.1
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs_abpo/_logs
STEPS=${STEPS:-5000000}
echo "=== norm+lam start $(date -Is)"
for seed in 1 2 3 4 5; do
  for spec in "Hopper-v4 Hopper-V4 0.999" "Humanoid-v4 Humanoid-V4 0.99" "Ant-v4 Ant-V4 0.99"; do
    set -- $spec
    out="runs_abpo/$1/bpo_norm_lam0.1/s$seed"
    if [ -f "$out/DONE" ]; then continue; fi
    ( python scripts/train_local.py --out "$out" \
        --sweep "sweeps/mujoco/$2/bpo_seed.yaml" training.seed=$seed training.total_timesteps=$STEPS \
        env.normalize=true env.norm_gamma=$3 algo.lambda_mode=adaptive algo.lambda_=0.1 \
        > "runs_abpo/_logs/$1-bpo_norm_lam0.1-s$seed.log" 2>&1 && touch "$out/DONE" ) &
  done
done
wait
echo "NORMLAM_DONE $(date -Is)"
