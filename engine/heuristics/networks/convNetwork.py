import torch.nn as nn

from engine.heuristics.networks.abstractNetwork import AbstractNetwork


class ConvNetwork(AbstractNetwork):
    def init_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(12, 12, (8, 8), padding=(7, 7)),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(12, 12, (8, 8), padding=(7, 7)),
            nn.MaxPool2d(2, padding=1),
            nn.Flatten(),
            nn.Linear(768, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def board_to_tensor(state):
        tensor = AbstractNetwork.board_to_tensor(state)
        return tensor.view(1, 12, 8, 8)
