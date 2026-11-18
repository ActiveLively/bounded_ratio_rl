#!/usr/bin/env bash
# Pilot 3, queued behind scripts/std_floor_pilot.sh (waits for its STD_PILOT_DONE line).
#   Humanoid-v4, 5 seeds, 5M steps, CPU:  bpo_norm / ppo_norm = reproduction settings + VecNormalize
#   Go1-rough, 1 seed, 1500 iterations:   online = plain BPO with online advantages (reproduction used offline)
# Finished runs are skipped. Compare with: python scripts/plot_abpo.py --variants bpo_norm ppo_norm online
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p runs_abpo/_logs
STEPS=${STEPS:-5000000}

echo "waiting for std-floor pilot $(date -Is)"
until grep -q STD_PILOT_DONE runs_std_pilot.log 2>/dev/null || ! tmux has-session -t stdfloor 2>/dev/null; do sleep 30; done
echo "=== pilot 3 start $(date -Is)"

for seed in 1 2 3 4 5; do
  for spec in "bpo 0.99" "ppo 0.98"; do   # norm_gamma = each sweep's algo.gamma
    set -- $spec
    python scripts/train_local.py --out "runs_abpo/Humanoid-v4/$1_norm/s$seed" \
      --sweep "sweeps/mujoco/Humanoid-V4/$1_seed.yaml" training.seed=$seed training.total_timesteps=$STEPS \
      env.normalize=true env.norm_gamma=$2 > "runs_abpo/_logs/Humanoid-v4-$1_norm-s$seed.log" 2>&1 &
  done
done

name="go1_rough-online-s1"
if ! ls logs/rsl_rl/unitree_go1_rough/*_"$name"/model_1500.pt >/dev/null 2>&1; then
  echo "=== $name start $(date -Is)"
  timeout 7200 python train_rsl_rl.py --headless --task=Isaac-Velocity-Rough-Unitree-Go1-v0 --algorithm bpo \
    --logger tensorboard --seed 1 --max_iterations 1500 --run_name "$name" \
    agent.algorithm.online_adv=True > "runs_abpo/_logs/$name.log" 2>&1
  echo "    exit=$? end=$(date -Is)"
fi

wait
echo "PILOT3_DONE $(date -Is)"
