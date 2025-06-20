import torch
import torch.nn as nn

from engine.heuristics.abstractPolicyNetwork import AbstractPolicyNetwork
from engine.heuristics.modules import GlobalSoftmax


class SimplePolicy(AbstractPolicyNetwork):
    def init_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(11, 73, (3, 3), padding=1),
            GlobalSoftmax(),
        )

    @staticmethod
    def board_to_tensor(state, data_type=torch.int8):
        tensor = AbstractPolicyNetwork.board_to_tensor(state, data_type)
        return tensor.view(1, 11, 8, 8)
