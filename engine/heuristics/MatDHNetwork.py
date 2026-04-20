from bulletchess import CHECKMATE, DRAW
import torch
import torch.nn as nn

from engine.heuristics.abstractPolicyNetwork import AbstractPolicyNetwork
from engine.heuristics.modules import ResidualBlock
from engine.values import OUTCOMES


class MaterialDualHeadNetwork(AbstractPolicyNetwork):
    def init_model(self):
        self.model = MaterialDualHeadModule()

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


class MaterialDualHeadModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(11, 16, kernel_size=3, padding=1),
            ResidualBlock(16, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )

        self.value_phase_one = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value_phase_two = nn.Sequential(
            nn.Linear(69, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(32, 73, kernel_size=1),
        )

    def forward(self, x):
        b = self.body(x)
        v1 = self.value_phase_one(b)
        mat = x[:, 0:5, :, :].sum(dim=(2, 3))
        y = torch.cat((v1, mat), dim=1)
        v2 = self.value_phase_two(y)
        policy = self.policy_head(b)
        return v2, policy
