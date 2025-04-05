import torch.nn as nn

from engine.heuristics.networks.abstractNetwork import AbstractNetwork


class BasicNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()

    def init_model(self):
        self.model = nn.Sequential(
            nn.Linear(768, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
