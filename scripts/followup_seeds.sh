#!/usr/bin/env bash
# Follow-up seeds for the two ideas that survived the pilots. Finished runs are skipped.
#   scripts/followup_seeds.sh lam    Hopper + Humanoid, seeds 1-10, 5M steps, CPU, starts at once:
#                                    lam_c0.1 = reproduction BPO settings with only lambda made scale-free
#                                    (lambda = 0.1 * mean|Q - mu|; fixed ratio bound, unlike abpo_c0.1)
#   scripts/followup_seeds.sh go1    Go1-rough std floor 0.2, seeds 2-5, GPU, waits for pilot 3's Go1 run
# Compare with: python scripts/plot_abpo.py --variants lam_c0.1 std0.2
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p runs_abpo/_logs
STEPS=${STEPS:-5000000}

case "${1:-}" in
lam)
  echo "=== lam_c0.1 start $(date -Is)"
  for seed in 1 2 3 4 5 6 7 8 9 10; do
    for pair in "Hopper-v4 Hopper-V4" "Humanoid-v4 Humanoid-V4"; do
      set -- $pair
      python scripts/train_local.py --out "runs_abpo/$1/lam_c0.1/s$seed" \
        --sweep "sweeps/mujoco/$2/bpo_seed.yaml" training.seed=$seed training.total_timesteps=$STEPS \
        algo.lambda_mode=adaptive algo.lambda_=0.1 > "runs_abpo/_logs/$1-lam_c0.1-s$seed.log" 2>&1 &
    done
  done
  wait
  echo "LAM_DONE $(date -Is)"
  ;;
go1)
  echo "waiting for pilot 3's Go1 run $(date -Is)"
  until grep -q "exit=" runs_pilot3.log 2>/dev/null || ! tmux has-session -t pilot3 2>/dev/null; do sleep 30; done
  for seed in 2 3 4 5; do
    name="go1_rough-std0.2-s$seed"
    if ls logs/rsl_rl/unitree_go1_rough/*_"$name"/model_1500.pt >/dev/null 2>&1; then continue; fi
    echo "=== $name start $(date -Is)"
    timeout 7200 python train_rsl_rl.py --headless --task=Isaac-Velocity-Rough-Unitree-Go1-v0 --algorithm bpo \
      --logger tensorboard --seed $seed --max_iterations 1500 --run_name "$name" \
      agent.algorithm.online_adv=False agent.policy.min_std=0.2 > "runs_abpo/_logs/$name.log" 2>&1
    echo "    exit=$? end=$(date -Is)"
  done
  echo "GO1_SEEDS_DONE $(date -Is)"
  ;;
*) echo "usage: $0 lam|go1"; exit 2 ;;
esac
