import glob
import time
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_class, instantiate
import ale_py

gym.register_envs(ale_py)


class VideoUploadCallback(BaseCallback):
    """Log each finished clip to wandb; monitor_gym doesn't hook SB3's recorder."""

    def __init__(self, video_dir: str):
        super().__init__()
        self.video_dir = video_dir
        self.uploaded = set()

    def upload_new(self) -> None:
        for path in sorted(glob.glob(f"{self.video_dir}/*.mp4")):
            if path not in self.uploaded:
                self.uploaded.add(path)
                wandb.log({"videos": wandb.Video(path, format="mp4")})

    def _on_rollout_end(self) -> None:
        self.upload_new()

    def _on_step(self) -> bool:
        return True


class StopAfterHours(BaseCallback):
    """Stop training gracefully once a wall-clock budget is used up."""

    def __init__(self, max_hours: float):
        super().__init__()
        self.deadline = time.time() + max_hours * 3600

    def _on_step(self) -> bool:
        return time.time() < self.deadline


class RefillReplayBuffer(BaseCallback):
    """After a resume, act with the loaded policy for `n_steps` without updates.

    The replay buffer isn't saved with the model, so this refills it before training
    continues instead of updating on a near-empty buffer.
    """

    def __init__(self, n_steps: int):
        super().__init__()
        self.n_steps = n_steps

    def _on_training_start(self) -> None:
        self.gradient_steps = self.model.gradient_steps
        self.until = self.model.num_timesteps + self.n_steps
        self.model.gradient_steps = 0

    def _on_step(self) -> bool:
        if self.model.gradient_steps == 0 and self.model.num_timesteps >= self.until:
            self.model.gradient_steps = self.gradient_steps
        return True


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg : DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))
    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    resume = cfg.training.get("resume")
    if resume:
        # Continue the same wandb run; the new total_timesteps replaces the old one in its config
        run = wandb.init(**wandb_cfg, id=resume, resume="must", sync_tensorboard=True)
        run.config.update(config, allow_val_change=True)
    else:
        wandb_cfg['config'] = config
        run = wandb.init(**wandb_cfg, sync_tensorboard=True )
    env = instantiate(cfg.env)

    algo_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    # A resumed run gets its own TensorBoard dir so wandb doesn't sync the old event files again
    algo_cfg['tensorboard_log'] = f"runs/{run.id}" + ("-resume" if resume else "")
    callbacks = []
    if resume:
        # load() resets the env on the next learn(); the optimizer and target net come from the save
        model = get_class(algo_cfg['_target_']).load(
            f"models/{resume}/model.zip", env=env, device=algo_cfg.get('device', 'auto'),
            tensorboard_log=algo_cfg['tensorboard_log'],
        )
        if hasattr(model, "replay_buffer"):
            callbacks.append(RefillReplayBuffer(model.learning_starts))
    else:
        model = instantiate(algo_cfg, env=env)
    done_steps = model.num_timesteps
    remaining = cfg.training.total_timesteps - done_steps
    if remaining <= 0:
        print(f"already at {done_steps} of {cfg.training.total_timesteps} steps, nothing to do")
        run.finish()
        return

    # One clip ending at 25%, 50%, 75% and 100% of training.
    # The recorder counts vectorized steps since this process started, i.e. env steps / n_envs.
    video_length = 300
    total_steps = cfg.training.total_timesteps // env.num_envs
    offset = done_steps // env.num_envs
    video_starts = {
        max(int(total_steps * frac) - video_length - offset, 0)
        for frac in (0.25, 0.5, 0.75, 1.0)
        if int(total_steps * frac) > offset
    }
    video_dir = f"videos/{run.id}"
    env = VecVideoRecorder(
        env,
        video_dir,
        record_video_trigger=lambda x: x in video_starts,
        video_length=video_length,
    )
    video_callback = VideoUploadCallback(video_dir)

    model.learn(
        total_timesteps=remaining,
        reset_num_timesteps=not resume,
        callback=callbacks + [
            WandbCallback(
                gradient_save_freq=cfg.training.gradient_save_freq,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
            video_callback,
        ] + ([StopAfterHours(cfg.training.max_hours)] if cfg.training.max_hours else []),
    )
    env.close()  # flushes the final clip if training stopped mid-recording
    video_callback.upload_new()
    run.finish()


if __name__ == "__main__":
    main()
