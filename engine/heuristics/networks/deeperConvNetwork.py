import torch.nn as nn

from engine.heuristics.networks.abstractNetwork import AbstractNetwork


class DeeperConvNetwork(AbstractNetwork):
    def init_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def board_to_tensor(state):
        tensor = AbstractNetwork.board_to_tensor(state)
        return tensor.view(1, 12, 8, 8)
