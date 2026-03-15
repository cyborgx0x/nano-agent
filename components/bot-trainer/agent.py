import torch
import torch.nn.functional as F
import torch.optim as optim

from policy import ComplexPolicyNetwork


class PPOAgent:
    def __init__(
        self, policy_net, input_dim, action_space, lr=3e-4, gamma=0.99, eps_clip=0.2
    ):
        self.policy_net = policy_net
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy_old = ComplexPolicyNetwork(input_dim, action_space)
        self.policy_old.load_state_dict(policy_net.state_dict())
        self.max_grad_norm = 0.5  # Gradient clipping threshold

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        dist, value = self.policy_old(state)
        if dist is None or value is None:
            return None, None, None
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def update(self, memory):
        rewards, discounted_reward = [], 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - is_terminal))
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)  # Reshape rewards to [batch_size, 1]
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.tensor(memory.states, dtype=torch.float32)
        old_actions = torch.tensor(memory.actions, dtype=torch.int64)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        old_values = torch.tensor(memory.values, dtype=torch.float32).unsqueeze(1)  # Reshape old_values to [batch_size, 1]

        for _ in range(4):
            self._optimize_policy(old_states, old_actions, old_logprobs, rewards, old_values)

        self.policy_old.load_state_dict(self.policy_net.state_dict())

    def _optimize_policy(self, old_states, old_actions, old_logprobs, rewards, old_values):
        try:
            dists, state_values = self.policy_net(old_states)
            if dists is None or state_values is None:
                print("Skipping optimization step due to NaNs")
                return

            logprobs = dists.log_prob(old_actions)
            dist_entropy = dists.entropy()

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - old_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            state_values = state_values.clamp(min=-1e6, max=1e6)
            rewards = rewards.clamp(min=-1e6, max=1e6)

            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
        except Exception as e:
            print(f"Error during optimization: {e}")
            print("Skipping this optimization step")
