# BPO reproduction results

Final training reward = mean of the last 100 training episodes at the end of each run (not a separate deterministic evaluation). Values are mean ± std over seeds. Pulled from wandb on 2026-10-28.

## MuJoCo

| Env | Algo | Seeds | Reward | Min | Max | Env steps |
|---|---|---|---|---|---|---|
| Ant-v4 | bpo | 10 | 5399.2 ± 740.8 | 3655.7 | 6130.7 | 40.0M |
| Ant-v4 | ppo | 10 | 3579.1 ± 1162.3 | 1517.8 | 5000.5 | 40.0M |
| Ant-v4 | sac | 10 | 5014.3 ± 1589.8 | 1339.3 | 6626.6 | 11.6M |
| Hopper-v4 | bpo | 10 | 3425.0 ± 280.3 | 2814.0 | 3716.5 | 40.0M |
| Hopper-v4 | bpo_lam0.001 | 5 | 2621.3 ± 932.1 | 1002.9 | 3250.4 | 40.0M |
| Hopper-v4 | ppo | 10 | 1182.0 ± 720.6 | 522.6 | 2562.8 | 40.0M |
| Hopper-v4 | sac | 10 | 2892.6 ± 665.9 | 2121.5 | 3784.4 | 11.7M |
| Humanoid-v4 | bpo | 10 | 1747.4 ± 1429.4 | 743.9 | 4431.1 | 40.0M |
| Humanoid-v4 | ppo | 10 | 990.3 ± 801.1 | 604.3 | 3252.4 | 40.0M |
| Humanoid-v4 | sac | 10 | 6930.2 ± 415.7 | 6120.1 | 7617.1 | 6.8M |
| Swimmer-v4 | bpo | 10 | 354.5 ± 3.5 | 349.8 | 360.1 | 40.0M |
| Swimmer-v4 | ppo | 10 | 250.2 ± 131.2 | 9.8 | 357.6 | 40.0M |
| Swimmer-v4 | sac | 10 | 122.9 ± 13.5 | 98.5 | 137.3 | 10.7M |

## Atari

| Env | Algo | Seeds | Reward | Min | Max | Env steps |
|---|---|---|---|---|---|---|
| AsterixNoFrameskip-v4 | bpo | 10 | 20519.2 ± 3626.3 | 14811.0 | 25464.5 | 40.0M |
| AsterixNoFrameskip-v4 | dqn | 10 | 3505.9 ± 252.8 | 3081.5 | 3906.5 | 5.1M |
| AsterixNoFrameskip-v4 | ppo | 10 | 6895.6 ± 1279.3 | 5056.0 | 8715.0 | 40.0M |
| BoxingNoFrameskip-v4 | bpo | 10 | 94.2 ± 3.1 | 90.2 | 98.5 | 20.0M |
| BoxingNoFrameskip-v4 | dqn | 10 | 92.1 ± 1.4 | 88.8 | 93.5 | 5.2M |
| BoxingNoFrameskip-v4 | ppo | 10 | 96.1 ± 2.5 | 92.9 | 99.4 | 20.0M |
| BreakoutNoFrameskip-v4 | bpo | 10 | 363.7 ± 24.5 | 303.4 | 388.1 | 20.0M |
| BreakoutNoFrameskip-v4 | dqn | 10 | 203.5 ± 10.2 | 185.0 | 220.8 | 5.1M |
| BreakoutNoFrameskip-v4 | ppo | 10 | 348.5 ± 17.9 | 321.0 | 371.7 | 20.0M |
| PongNoFrameskip-v4 | bpo | 10 | 20.6 ± 0.3 | 20.0 | 20.9 | 20.0M |
| PongNoFrameskip-v4 | dqn | 10 | 20.4 ± 0.1 | 20.2 | 20.7 | 5.3M |
| PongNoFrameskip-v4 | ppo | 10 | 20.6 ± 0.3 | 20.1 | 20.9 | 20.0M |

## IsaacLab

| Env | Algo | Seeds | Reward | Min | Max | Terrain level |
|---|---|---|---|---|---|---|
| anymal_c_rough | bpo | 5 | 15.9 ± 0.8 | 15.3 | 17.2 | 5.49 |
| anymal_c_rough | ppo | 5 | 16.8 ± 0.6 | 15.8 | 17.4 | 5.98 |
| g1_rough | bpo | 5 | 35.2 ± 3.5 | 29.8 | 38.1 | 5.78 |
| g1_rough | bpo_lam0.01 | 5 | 36.4 ± 1.7 | 35.1 | 39.2 | 5.82 |
| g1_rough | bpo_lam0.03 | 5 | 32.1 ± 1.6 | 30.4 | 34.0 | 5.83 |
| g1_rough | bpo_lam0.1 | 5 | 2.7 ± 9.4 | -3.7 | 19.3 | 1.06 |
| g1_rough | bpo_rkl | 5 | -1.3 ± 0.6 | -2.0 | -0.8 | 0.20 |
| g1_rough | bpo_tv | 5 | 24.1 ± 5.6 | 14.3 | 28.2 | 5.50 |
| g1_rough | bpo_tvcoef0.1 | 5 | 32.3 ± 2.0 | 30.1 | 35.3 | 5.62 |
| g1_rough | bpo_tvcoef0.3 | 5 | 31.9 ± 2.9 | 28.4 | 34.8 | 5.39 |
| g1_rough | bpo_tvcoef1.0 | 5 | 25.5 ± 5.4 | 21.4 | 32.1 | 5.25 |
| g1_rough | bpo_tvcoef10.0 | 5 | 26.1 ± 3.9 | 20.7 | 30.6 | 5.20 |
| g1_rough | bpo_tvcoef3.0 | 5 | 28.0 ± 1.8 | 25.5 | 30.3 | 5.43 |
| g1_rough | bpo_value | 5 | 34.2 ± 2.2 | 31.7 | 37.2 | 5.74 |
| g1_rough | ppo | 5 | 25.5 ± 1.1 | 23.9 | 26.8 | 5.81 |
| go1_rough | bpo | 5 | 19.2 ± 1.6 | 17.5 | 21.4 | 1.00 |
| go1_rough | ppo | 5 | 23.2 ± 0.9 | 21.8 | 24.1 | 6.04 |
| h1_rough | bpo | 5 | 30.1 ± 1.3 | 28.4 | 32.0 | 5.69 |
| h1_rough | ppo | 5 | 28.1 ± 0.5 | 27.6 | 28.7 | 5.82 |
