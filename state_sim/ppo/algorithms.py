from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Transition:
    obs: list[float]
    action: int
    logprob: float
    value: float
    reward: float
    done: float
    hidden: list[float] | None = None


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_adv = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + values
    return advantages, returns
