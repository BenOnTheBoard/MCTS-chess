import torch.nn as nn

from engine.heuristics.networks.abstractNetwork import AbstractNetwork


class ConvNetwork(AbstractNetwork):
    def init_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def board_to_tensor(state):
        tensor = AbstractNetwork.board_to_tensor(state)
        return tensor.view(1, 6, 8, 8)
