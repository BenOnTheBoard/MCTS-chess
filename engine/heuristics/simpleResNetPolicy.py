import torch
import torch.nn as nn

from engine.heuristics.abstractPolicyNetwork import AbstractPolicyNetwork
from engine.heuristics.modules import ResidualBlock


class SimpleResNetPolicy(AbstractPolicyNetwork):
    def init_model(self):
        self.model = nn.Sequential(
            ResidualBlock(11, 16),
            ResidualBlock(16, 32),
            ResidualBlock(32, 64),
            nn.Conv2d(64, 73, kernel_size=1, bias=False),
        )

    @staticmethod
    def board_to_tensor(state, data_type=torch.int8):
        tensor = AbstractPolicyNetwork.board_to_tensor(state, data_type)
        return tensor.view(1, 11, 8, 8)
