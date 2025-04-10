import torch.nn as nn

from engine.heuristics.networks.abstractNetwork import AbstractNetwork


class MinNetwork(AbstractNetwork):
    def init_model(self):
        self.model = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid(),
        )
