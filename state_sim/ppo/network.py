from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return residual + x


class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Feedforward Actor-Critic network.
    Compact design: small obs (no memory, no world model) → fast training.
    hidden_dim=128, 1 expert, 1 residual block.
    """

    def __init__(self, obs_dim: int, action_dim: int, memory_size: int = 0):
        super().__init__()
        hidden_dim = 128
        expert_hidden = 192
        n_experts = 1

        self.input_norm = nn.LayerNorm(obs_dim)
        self.stem = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.experts = nn.ModuleList(
            [Expert(hidden_dim, expert_hidden) for _ in range(n_experts)]
        )
        self.expert_gate = nn.Linear(hidden_dim, n_experts)

        self.post_expert = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=0.1),
        )

        self.policy_tower = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.value_tower = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def init_hidden(
        self, batch_size: int = 1
    ) -> None:
        """Dummy method for backward compatibility. Returns None."""
        _ = batch_size  # Unused, kept for compatibility
        return None

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Forward pass (feedforward, no recurrence).

        Args:
            x: observation tensor [batch, obs_dim]
            hidden: ignored (kept for backward compatibility)

        Returns:
            (policy_logits, value, None)
        """
        h = self.stem(self.input_norm(x))

        # Mixture of Experts
        gate = torch.softmax(self.expert_gate(h), dim=-1)
        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=1)
        mixed = (gate.unsqueeze(-1) * expert_outs).sum(dim=1)

        # Residual blocks (no GRU)
        h = self.post_expert(h + mixed)

        return self.policy_tower(h), self.value_tower(h), None
