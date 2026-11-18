#!/usr/bin/env bash
# Go1-rough: BPO with online advantages AND action-std floor 0.2, seed 1, 1500 iterations.
# Compare with: python scripts/plot_abpo.py --variants online std0.2 online_std0.2
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p runs_abpo/_logs
for seed in ${SEEDS:-1}; do
  name="go1_rough-online_std0.2-s$seed"
  if ls logs/rsl_rl/unitree_go1_rough/*_"$name"/model_1500.pt >/dev/null 2>&1; then continue; fi
  echo "=== $name start $(date -Is)"
  timeout 7200 python train_rsl_rl.py --headless --task=Isaac-Velocity-Rough-Unitree-Go1-v0 --algorithm bpo \
    --logger tensorboard --seed $seed --max_iterations 1500 --run_name "$name" \
    agent.algorithm.online_adv=True agent.policy.min_std=0.2 > "runs_abpo/_logs/$name.log" 2>&1
  echo "    exit=$? end=$(date -Is)"
done
echo "COMBO_DONE $(date -Is)"
