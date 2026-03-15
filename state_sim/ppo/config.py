from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOTrainConfig:
    episodes: int = 1000
    max_steps: int = 600  # shorter episodes → faster training iteration
    seed: int = 42
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_eps: float = 0.15
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    update_epochs: int = 4
    minibatch_size: int = 1024
    update_every_episodes: int = 10
    log_interval: int = 20
    use_wandb: bool = True
    wandb_project: str = "nano-agent-state-sim"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    checkpoint_dir: str = "runs/state_sim"
    objective: str = "navigation"
    mixed_navigation_ratio: float = 0.5
    nav_teacher_start: float = 0.40
    nav_teacher_min: float = 0.05
    gather_teacher_start: float = 0.80
    gather_teacher_min: float = 0.35
    gather_teacher_decay_power: float = 2.0
    gather_teacher_recovery_boost: float = 0.20
    gather_teacher_recovery_ratio: float = 0.75
    gather_teacher_boost_decay: float = 0.96
    mixed_teacher_start: float = 0.50
    mixed_teacher_min: float = 0.10
    gather_focus_zone: str = "forest_cross"
    gather_focus_resource_scale: float = 1.8
    gather_train_mode: str = "multi"
    gather_zone_scheduler: str = "round_robin"
    gather_holdout_ratio: float = 0.2
    gather_eval_interval: int = 30
    gather_eval_episodes: int = 30
    inventory_full_target: int = 20  # reduced: 20 gathers = inventory full (faster iteration)
    memory_size: int = 0  # no memory; kept for checkpoint compatibility
    # Vectorized env settings
    num_envs: int = 8           # parallel envs → GPU batch [N, obs_dim] per step
    rollout_steps: int = 2048   # steps per env per rollout; total buffer = num_envs * rollout_steps
