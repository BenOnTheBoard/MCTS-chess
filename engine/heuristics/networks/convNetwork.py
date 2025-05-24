import torch.nn as nn

from engine.heuristics.networks.abstractNetwork import AbstractNetwork


class ConvNetwork(AbstractNetwork):
    def init_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(11, 16, (3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def board_to_tensor(state):
        tensor = AbstractNetwork.board_to_tensor(state)
        return tensor.view(1, 11, 8, 8)
