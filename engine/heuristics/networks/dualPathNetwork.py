import torch
import torch.nn as nn

from engine.heuristics.networks.abstractNetwork import AbstractNetwork


class DualPathModel(nn.Module):
    def __init__(self):
        super(DualPathModel, self).__init__()

        self.cnn_path = nn.Sequential(
            nn.Conv2d(6, 16, (3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

        self.material_values = nn.Parameter(torch.randn(6))

        self.fc_layers = nn.Sequential(
            nn.Linear(134, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        cnn_out = self.cnn_path(x)
        piece_counts = x.sum(dim=[2, 3])

        material_scores = piece_counts * self.material_values
        combined = torch.cat([cnn_out, material_scores], dim=1)

        return self.fc_layers(combined)


class DualPathNetwork(AbstractNetwork):
    def init_model(self):
        self.model = DualPathModel()

    @staticmethod
    def board_to_tensor(state):
        tensor = AbstractNetwork.board_to_tensor(state)
        return tensor.view(1, 6, 8, 8)


if __name__ == "__main__":
    import chess

    b = chess.Board()
    dpn = DualPathNetwork()
    print(dpn.evaluate(b))
