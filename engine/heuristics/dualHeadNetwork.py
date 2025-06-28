from bulletchess import CHECKMATE, DRAW
import torch
import torch.nn as nn

from engine.heuristics.abstractPolicyNetwork import AbstractPolicyNetwork
from engine.heuristics.modules import ResidualBlock
from engine.values import OUTCOMES


class DualHeadNetwork(AbstractPolicyNetwork):
    def init_model(self):
        self.model = DualHeadModule()

    @staticmethod
    def board_to_tensor(state, data_type=torch.int8):
        tensor = AbstractPolicyNetwork.board_to_tensor(state, data_type)
        return tensor.view(1, 11, 8, 8)

    def evaluate(self, state):
        if state in CHECKMATE:
            return -OUTCOMES[state.turn], None
        if state in DRAW:
            return OUTCOMES[None], None

        value, policy = self.tensor_eval(state)
        return value.item(), policy


class DualHeadModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(11, 16, kernel_size=3, padding=1),
            ResidualBlock(16, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )  # 64x8x8

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 73, kernel_size=1),
        )

    def forward(self, x):
        x = self.body(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy
