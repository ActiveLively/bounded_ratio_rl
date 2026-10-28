import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO


def record_ratio_stats(model) -> None:
    """
    Log the mean ratio pi(a|s)/pi_old(a|s) over the rollout buffer after an update,
    split into ratios above and below 1.0 (ratio statistics, Fig. 7 of the paper).
    """
    with th.no_grad():
        rollout_data = next(model.rollout_buffer.get(batch_size=None))
        actions = rollout_data.actions
        if isinstance(model.action_space, spaces.Discrete):
            actions = actions.long().flatten()
        log_prob = model.policy.get_distribution(rollout_data.observations).log_prob(actions)
        ratio = th.exp(log_prob.flatten() - rollout_data.old_log_prob.flatten())

    high, low = ratio[ratio > 1], ratio[ratio < 1]
    model.logger.record("train/ratio_high", high.mean().item() if high.numel() else 1.0)
    model.logger.record("train/ratio_low", low.mean().item() if low.numel() else 1.0)


class RatioStatsPPO(PPO):
    """Stock SB3 PPO that additionally logs ratio statistics after each update."""

    def train(self) -> None:
        super().train()
        record_ratio_stats(self)
