# Adapted from Meta's V-JEPA for game control
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for game agent use case

import numpy as np
import torch
import torch.nn.functional as F


class GameWorldModel:
    """
    World model for game agent control.
    Adapted from V-JEPA's WorldModel for mouse/keyboard actions instead of robot poses.
    """

    def __init__(
        self,
        encoder,
        predictor,
        tokens_per_frame,
        transform,
        mpc_args={
            "rollout": 3,           # Look ahead 3 steps
            "samples": 200,         # Sample 200 action sequences
            "topk": 10,            # Keep top 10 best sequences
            "cem_steps": 10,       # 10 optimization iterations
            "momentum_mean": 0.15,
            "momentum_std": 0.15,
            "max_mouse_move": 100,  # Max pixels to move mouse per step
            "verbose": False,
        },
        normalize_reps=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            encoder: Vision encoder (e.g., ViT)
            predictor: Latent predictor
            tokens_per_frame: Number of visual tokens per frame
            transform: Image preprocessing transform
            mpc_args: Model Predictive Control parameters
            normalize_reps: Whether to normalize representations
            device: torch device
        """
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.transform = transform
        self.tokens_per_frame = tokens_per_frame
        self.device = device
        self.mpc_args = mpc_args

    def encode(self, image):
        """
        Encode game screenshot to latent representation.

        Args:
            image: numpy array [H, W, C] (RGB screenshot)

        Returns:
            latent: torch tensor [1, N_tokens, D] (latent representation)
        """
        # Add batch and time dimensions
        clip = np.expand_dims(image, axis=0)  # [1, H, W, C]
        clip = self.transform(clip)[None, :]  # [1, 1, C, H, W]

        B, C, T, H, W = clip.size()
        # Prepare for encoder (expects video input)
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(self.device, non_blocking=True)

        # Encode
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)

        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))

        return h

    def predict_next_state(self, current_rep, action):
        """
        Predict next latent state given current state and action.

        Args:
            current_rep: [B, N_tokens, D] current latent representation
            action: [B, 1, action_dim] action to take (mouse_x, mouse_y, click, key)

        Returns:
            next_rep: [B, N_tokens, D] predicted next latent
        """
        B, N_T, D = current_rep.size()

        # Use predictor
        next_rep = self.predictor(current_rep, action)[:, -self.tokens_per_frame:]

        if self.normalize_reps:
            next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))

        return next_rep

    def plan_action(self, current_rep, goal_rep, current_mouse_pos=None):
        """
        Plan next action using Model Predictive Control (MPC).
        Uses Cross-Entropy Method (CEM) to find best action sequence.

        Args:
            current_rep: [1, N_tokens, D] current state representation
            goal_rep: [1, N_tokens, D] goal state representation
            current_mouse_pos: [x, y] current mouse position (optional)

        Returns:
            action: dict with keys 'mouse_pos', 'click', 'key'
        """
        if current_mouse_pos is None:
            current_mouse_pos = torch.zeros(1, 1, 2, device=self.device)
        else:
            current_mouse_pos = torch.tensor(current_mouse_pos, device=self.device).view(1, 1, 2)

        # Define step predictor for MPC
        def step_predictor(reps, actions, mouse_positions):
            """
            Args:
                reps: [B, T, N_T, D] representation trajectory
                actions: [B, T, action_dim] action trajectory
                mouse_positions: [B, T, 2] mouse position trajectory
            """
            B, T, N_T, D = reps.size()
            reps_flat = reps.flatten(1, 2)

            # Predict next representation
            next_rep = self.predictor(reps_flat, actions)[:, -self.tokens_per_frame:]

            if self.normalize_reps:
                next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))

            next_rep = next_rep.view(B, 1, N_T, D)

            # Compute new mouse position (integrate velocity)
            next_mouse_pos = mouse_positions[:, -1:] + actions[:, -1:, :2]

            return next_rep, next_mouse_pos

        # Run CEM optimization
        best_action = self._cem_optimize(
            current_rep=current_rep,
            current_mouse_pos=current_mouse_pos,
            goal_rep=goal_rep,
            step_predictor=step_predictor,
        )

        # Convert to action dict
        action = {
            'mouse_delta': best_action[0, :2].cpu().numpy(),  # [dx, dy]
            'click': best_action[0, 2].item() > 0.5,  # Binary click
            'key': None,  # TODO: Add key press support
        }

        return action

    def _cem_optimize(self, current_rep, current_mouse_pos, goal_rep, step_predictor):
        """
        Cross-Entropy Method optimization for action planning.
        Simplified version adapted for game control.
        """
        rollout = self.mpc_args['rollout']
        samples = self.mpc_args['samples']
        topk = self.mpc_args['topk']
        cem_steps = self.mpc_args['cem_steps']
        max_mouse_move = self.mpc_args['max_mouse_move']

        # Replicate for batch processing
        current_rep = current_rep.repeat(samples, 1, 1, 1)  # [S, 1, N_T, D]
        goal_rep = goal_rep.repeat(samples, 1, 1, 1)
        current_mouse_pos = current_mouse_pos.repeat(samples, 1, 1)  # [S, 1, 2]

        # Initialize action distribution
        # Action: [dx, dy, click] (3D)
        mean = torch.zeros((rollout, 3), device=self.device)
        std = torch.ones((rollout, 3), device=self.device)
        std[:, :2] = max_mouse_move  # Mouse movement range
        std[:, 2] = 0.5  # Click probability

        # CEM iterations
        for step in range(cem_steps):
            # Sample action trajectories
            action_trajs = []
            rep_trajs = [current_rep]
            mouse_pos_traj = current_mouse_pos

            for h in range(rollout):
                # Sample actions from current distribution
                actions = torch.randn(samples, 3, device=self.device) * std[h] + mean[h]

                # Clip to valid ranges
                actions[:, :2] = torch.clamp(actions[:, :2], -max_mouse_move, max_mouse_move)
                actions[:, 2] = torch.sigmoid(actions[:, 2])  # Click probability

                actions = actions.unsqueeze(1)  # [S, 1, 3]
                action_trajs.append(actions)

                # Predict next state
                next_rep, next_mouse_pos = step_predictor(
                    torch.stack(rep_trajs, dim=1),
                    torch.cat(action_trajs, dim=1),
                    mouse_pos_traj
                )

                rep_trajs.append(next_rep)
                mouse_pos_traj = torch.cat([mouse_pos_traj, next_mouse_pos], dim=1)

            # Evaluate trajectories
            final_reps = rep_trajs[-1]  # [S, 1, N_T, D]
            action_traj = torch.cat(action_trajs, dim=1)  # [S, rollout, 3]

            # Compute similarity to goal (L1 distance)
            distances = torch.mean(
                torch.abs(final_reps.flatten(1) - goal_rep.flatten(1)),
                dim=-1
            )  # [S]

            # Select top-k trajectories
            topk_indices = torch.topk(distances, topk, largest=False).indices
            best_actions = action_traj[topk_indices]  # [topk, rollout, 3]

            # Update distribution
            mean = best_actions.mean(dim=0)  # [rollout, 3]
            std = best_actions.std(dim=0) + 1e-6  # [rollout, 3]

        # Return first action from best trajectory
        return mean[0]  # [3]


def compute_action_delta(prev_mouse_pos, new_mouse_pos, click=False):
    """
    Compute action representation from mouse positions.

    Args:
        prev_mouse_pos: [x, y] previous mouse position
        new_mouse_pos: [x, y] new mouse position
        click: bool whether to click

    Returns:
        action: [dx, dy, click] action vector
    """
    delta = np.array(new_mouse_pos) - np.array(prev_mouse_pos)
    action = np.concatenate([delta, [1.0 if click else 0.0]])
    return action
