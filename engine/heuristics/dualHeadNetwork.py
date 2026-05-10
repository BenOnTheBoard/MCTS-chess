import torch
import torch.nn as nn

from engine.heuristics.abstractPolicyNetwork import AbstractPolicyNetwork
from engine.heuristics.modules import ResidualBlock


class DualHeadNetwork(AbstractPolicyNetwork):
    def init_model(self):
        self.model = DualHeadModule()

    @staticmethod
    def board_to_tensor(state, data_type=torch.int8):
        tensor = AbstractPolicyNetwork.board_to_tensor(state, data_type)
        return tensor.view(1, 13, 8, 8)

    def evaluate(self, state):
        value, policy = self.tensor_eval(state)
        return value.item(), policy


class DualHeadModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(13, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(16, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            ResidualBlock(64, 64),
            nn.Conv2d(64, 73, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.body(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy
