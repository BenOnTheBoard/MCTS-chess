import torch
import torch.nn as nn

from engine.heuristics.networks.abstractNetwork import AbstractNetwork
from engine.heuristics.networks.chessConvs import ChessConv2d


class DeltaOneModel(nn.Module):
    def __init__(self):
        super(DeltaOneModel, self).__init__()

        self.cnn_path = nn.Sequential(
            ChessConv2d(11, 16, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ChessConv2d(16, 32, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

        self.material_values = nn.Parameter(torch.randn(6))

        self.fc_layers = nn.Sequential(
            nn.Linear(139, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        cnn_out = self.cnn_path(x)
        piece_counts = x.sum(dim=[2, 3])[:, :6]
        other_information = x[:, 6:11, 0, 0]

        material_scores = piece_counts * self.material_values
        combined = torch.cat([cnn_out, material_scores, other_information], dim=1)

        return self.fc_layers(combined)


class DeltaOne(AbstractNetwork):
    def init_model(self):
        self.model = DeltaOneModel()

    @staticmethod
    def board_to_tensor(state, data_type=torch.int8):
        tensor = AbstractNetwork.board_to_tensor(state, data_type)
        return tensor.view(1, 11, 8, 8)
