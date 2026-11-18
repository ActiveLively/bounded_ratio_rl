#!/usr/bin/env bash
# Go1-rough, plain BPO with online advantages, seeds 2-5 (seed 1 ran in scripts/next_pilot.sh).
# Finished runs are skipped. Compare with: python scripts/plot_abpo.py --variants online std0.2
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p runs_abpo/_logs
for seed in 2 3 4 5; do
  name="go1_rough-online-s$seed"
  if ls logs/rsl_rl/unitree_go1_rough/*_"$name"/model_1500.pt >/dev/null 2>&1; then continue; fi
  echo "=== $name start $(date -Is)"
  timeout 7200 python train_rsl_rl.py --headless --task=Isaac-Velocity-Rough-Unitree-Go1-v0 --algorithm bpo \
    --logger tensorboard --seed $seed --max_iterations 1500 --run_name "$name" \
    agent.algorithm.online_adv=True > "runs_abpo/_logs/$name.log" 2>&1
  echo "    exit=$? end=$(date -Is)"
done
echo "ONLINE_SEEDS_DONE $(date -Is)"
