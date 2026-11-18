#!/usr/bin/env bash
# Reverse ablation of the Go1 finding. The reproduced ANYmal-C and G1 BPO baselines already use online
# advantages (the default; only the Go1 and H1 sweeps set online_adv=false), so here BPO is run with
# OFFLINE advantages on those two tasks to see whether it causes the same action-std collapse as on Go1.
# 3 seeds each, baseline iteration counts (ANYmal-C 1500, G1 3000). Finished runs are skipped.
# Compare with: python scripts/plot_abpo.py --variants offline
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p runs_abpo/_logs
for spec in "anymal_c_rough Isaac-Velocity-Rough-Anymal-C-v0 1500" "g1_rough Isaac-Velocity-Rough-G1-v0 3000"; do
  set -- $spec
  for seed in ${SEEDS:-1 2 3}; do
    name="$1-offline-s$seed"
    if ls logs/rsl_rl/$1/*_"$name"/model_$3.pt >/dev/null 2>&1; then continue; fi
    echo "=== $name start $(date -Is)"
    timeout 18000 python train_rsl_rl.py --headless --task=$2 --algorithm bpo \
      --logger tensorboard --seed $seed --max_iterations $3 --run_name "$name" \
      agent.algorithm.online_adv=False > "runs_abpo/_logs/$name.log" 2>&1
    echo "    exit=$? end=$(date -Is)"
  done
done
echo "ISAAC_OFFLINE_DONE $(date -Is)"
