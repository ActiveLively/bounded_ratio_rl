"""
Local SB3 training without wandb or video: same Hydra configs as train.py, logs to CSV + TensorBoard.

    python scripts/train_local.py --out runs_abpo/Hopper-v4/abpo/s1 \
        env=mujoco algo=bpo/mujoco env.env_id=Hopper-v4 training.seed=1 algo.adaptive_clip=true

Everything after --out is a Hydra override, exactly as for train.py. A sweep file from sweeps/
can be given with --sweep; its single-valued parameters become overrides (seed comes from the
command line).
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from hydra import compose, initialize_config_dir  # noqa: E402

try:  # Atari envs need explicit registration in gymnasium >= 1.0 (train.py does the same)
    import ale_py  # noqa: E402
    import gymnasium as _gym  # noqa: E402
    _gym.register_envs(ale_py)
except ImportError:
    pass
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402


class StopAfterHours(BaseCallback):
    def __init__(self, max_hours: float):
        super().__init__()
        self.deadline = time.time() + max_hours * 3600

    def _on_step(self) -> bool:
        return time.time() < self.deadline


def sweep_overrides(path: str) -> list[str]:
    """Single-valued parameters of a wandb sweep file as Hydra overrides (seed excluded)."""
    with open(path) as f:
        params = yaml.safe_load(f)["parameters"]
    out = []
    for key, spec in params.items():
        values = spec.get("values", [spec.get("value")])
        if key == "training.seed" or len(values) != 1:
            continue
        out.append(f"{key}={values[0]}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--sweep", default=None)
    parser.add_argument("--threads", type=int, default=1)
    args, overrides = parser.parse_known_args()
    torch.set_num_threads(args.threads)

    if args.sweep:
        overrides = sweep_overrides(args.sweep) + overrides
    with initialize_config_dir(config_dir=os.path.join(ROOT, "cfgs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    done_file = os.path.join(args.out, "done.json")
    if os.path.exists(done_file):
        print("already finished:", args.out)
        return

    env = instantiate(cfg.env)
    algo_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    algo_cfg.setdefault("device", "cpu")
    algo_cfg["seed"] = cfg.training.seed
    model = instantiate(algo_cfg, env=env)
    model.set_logger(configure(args.out, ["csv", "tensorboard"]))

    callbacks = [StopAfterHours(cfg.training.max_hours)] if cfg.training.max_hours else []
    start = time.time()
    model.learn(total_timesteps=cfg.training.total_timesteps, callback=callbacks, log_interval=10)
    env.close()
    with open(done_file, "w") as f:
        json.dump({"steps": int(model.num_timesteps), "seconds": time.time() - start}, f)


if __name__ == "__main__":
    main()
