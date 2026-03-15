from torch import nn
from torch.distributions import Categorical
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ComplexPolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_space):
        super(ComplexPolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, action_space)
        self.value_head = nn.Linear(64, 1)

        self.apply(init_weights)  # Apply weight initialization

    def forward(self, x):
        x = self.fc(x)

        if torch.isnan(x).any():
            print("Warning: NaNs detected after fully connected layers")
            return None, None

        action_logits = self.action_head(x)
        state_values = self.value_head(x)

        if torch.isnan(action_logits).any() or torch.isnan(state_values).any():
            print("Warning: NaNs detected in output layers")
            return None, None

        return Categorical(logits=action_logits), state_values
