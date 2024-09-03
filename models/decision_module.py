import torch
import torch.nn as nn

class DecisionModule(nn.Module):
    def __init__(self):
        super(DecisionModule, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Example output: steering angle and throttle
        )
        self.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Example value function
        )

    def forward(self, fused_features):
        policy = self.policy_net(fused_features)
        value = self.value_net(fused_features)
        return policy, value
