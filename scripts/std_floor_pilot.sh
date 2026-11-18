#!/usr/bin/env bash
# Pilot: does a floor on the action std fix BPO's Go1-rough plateau?
# Plain BPO (fixed eps and lambda, offline advantages as in the reproduction), 1 seed, 1500 iterations.
#   std0.2 / std0.3   action std clamped at 0.2 / 0.3 (it collapsed to ~0.1 in the adaptive-eps pilot)
# Finished runs are skipped. Compare with: python scripts/plot_abpo.py --variants std0.2 std0.3
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p runs_abpo/_logs

go1() {  # variant overrides...
  local variant=$1; shift
  local name="go1_rough-$variant-s1"
  if ls logs/rsl_rl/unitree_go1_rough/*_"$name"/model_1500.pt >/dev/null 2>&1; then return; fi
  echo "=== $name start $(date -Is)"
  timeout 7200 python train_rsl_rl.py --headless --task=Isaac-Velocity-Rough-Unitree-Go1-v0 --algorithm bpo \
    --logger tensorboard --seed 1 --max_iterations 1500 --run_name "$name" \
    agent.algorithm.online_adv=False "$@" > "runs_abpo/_logs/$name.log" 2>&1
  echo "    exit=$? end=$(date -Is)"
}
go1 std0.2 agent.policy.min_std=0.2
go1 std0.3 agent.policy.min_std=0.3

echo "STD_PILOT_DONE $(date -Is)"
