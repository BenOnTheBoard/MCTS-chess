import torch
import torch.nn as nn

from engine.heuristics.networks.abstractPolicyNetwork import AbstractPolicyNetwork


class GlobalSoftmax(nn.Module):
    def forward(self, x):
        og_shape = x.shape
        x_flat = x.view(-1)
        x_soft = torch.nn.functional.softmax(x_flat, dim=0)
        return x_soft.view(og_shape)


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
