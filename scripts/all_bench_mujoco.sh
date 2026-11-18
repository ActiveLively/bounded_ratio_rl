#!/usr/bin/env bash
# Fill in the MuJoCo tasks the earlier batches skipped. 5 seeds, 5M steps, CPU. Finished runs are skipped.
#   Swimmer-v4: bpo_norm, ppo_norm, bpo_norm_lam0.1, lam_c0.1      Ant-v4: lam_c0.1
#   *_norm = VecNormalize (norm_gamma = algo.gamma); lam = scale-free lambda (0.1 * mean|Q - mu|)
# Compare with: python scripts/plot_abpo.py --variants bpo_norm ppo_norm bpo_norm_lam0.1 lam_c0.1
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs_abpo/_logs
STEPS=${STEPS:-5000000}
NORM="env.normalize=true"
LAM="algo.lambda_mode=adaptive algo.lambda_=0.1"
run() {  # env sweepdir algo variant overrides...
  local env=$1 dir=$2 algo=$3 variant=$4; shift 4
  local out="runs_abpo/$env/$variant/s$seed"
  if [ -f "$out/DONE" ]; then return; fi
  ( python scripts/train_local.py --out "$out" --sweep "sweeps/mujoco/$dir/${algo}_seed.yaml" \
      training.seed=$seed training.total_timesteps=$STEPS "$@" \
      > "runs_abpo/_logs/$env-$variant-s$seed.log" 2>&1 && touch "$out/DONE" ) &
}
echo "=== all-bench mujoco start $(date -Is)"
for seed in 1 2 3 4 5; do
  run Swimmer-v4 Swimmer-V4 bpo bpo_norm        $NORM env.norm_gamma=0.9999
  run Swimmer-v4 Swimmer-V4 ppo ppo_norm        $NORM env.norm_gamma=0.9999
  run Swimmer-v4 Swimmer-V4 bpo bpo_norm_lam0.1 $NORM env.norm_gamma=0.9999 $LAM
  run Swimmer-v4 Swimmer-V4 bpo lam_c0.1        $LAM
  run Ant-v4     Ant-V4     bpo lam_c0.1        $LAM
done
wait
echo "MJ_ALL_DONE $(date -Is)"
