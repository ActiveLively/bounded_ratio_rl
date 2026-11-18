#!/usr/bin/env bash
# Pilot for Adaptive BPO: Go1 (1 seed, 2 settings) on the GPU, Hopper + Humanoid (3 seeds, 5M steps) on CPU.
#   abpo_eps   KL-controlled ratio bound only
#   abpo_c0.1  KL-controlled ratio bound + scale-free lambda = 0.1 * mean|Q - mu|
#   abpo_c1    same with lambda = 1.0 * mean|Q - mu|
# Baseline hyperparameters are the sweep files the reproduction used. Finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p runs_abpo/_logs
STEPS=${STEPS:-5000000}

mujoco() {  # env sweep-dir variant seed overrides...
  local env=$1 sweep=$2 variant=$3 seed=$4; shift 4
  python scripts/train_local.py --out "runs_abpo/$env/$variant/s$seed" \
    --sweep "sweeps/mujoco/$sweep/bpo_seed.yaml" training.seed=$seed training.total_timesteps=$STEPS "$@" \
    > "runs_abpo/_logs/$env-$variant-s$seed.log" 2>&1 &
}
for seed in 1 2 3; do
  for pair in "Hopper-v4 Hopper-V4" "Humanoid-v4 Humanoid-V4"; do
    set -- $pair
    mujoco $1 $2 abpo_eps  $seed algo.adaptive_clip=true
    mujoco $1 $2 abpo_c0.1 $seed algo.adaptive_clip=true algo.lambda_mode=adaptive algo.lambda_=0.1
    mujoco $1 $2 abpo_c1   $seed algo.adaptive_clip=true algo.lambda_mode=adaptive algo.lambda_=1.0
  done
done

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
go1 abpo_eps  agent.algorithm.clip_schedule=adaptive
go1 abpo_c0.1 agent.algorithm.clip_schedule=adaptive agent.algorithm.reg_w_mode=adaptive agent.algorithm.reg_w=0.1

wait
echo "PILOT_DONE $(date -Is)"
