import chess
import torch
import torch.nn as nn

from engine.heuristics.heuristicInterface import HeuristicInterface
from engine.values import PIECE_VALUES


class BasicNetwork(HeuristicInterface):
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = torch.nn.Sequential(
                nn.Conv3d(1, 1, (14, 8, 8), padding=(2, 7, 7)),
                nn.MaxPool2d(2, padding=1),
                nn.Conv3d(1, 1, (3, 8, 8), padding=(0, 7, 7)),
                nn.MaxPool2d(2, padding=1),
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

    def int_to_bit_vector(self, num):
        bit_list = [int(bit) for bit in bin(num)[2:].zfill(64)]
        bit_vector = torch.tensor(bit_list, dtype=torch.float)
        return bit_vector

    def tensor_eval(self, state):
        input_sections = []
        for colour in chess.COLORS:
            for piece in chess.PIECE_TYPES:
                pc_int = state.pieces_mask(piece, colour)
                section = self.int_to_bit_vector(pc_int)
                input_sections.append(section)

        input_vector = torch.concatenate(input_sections).view(1, 12, 8, 8)
        output_vector = self.model(input_vector)
        return output_vector

    def evaluate(self, state):
        if state.is_game_over():
            winner = state.outcome().winner
            if winner == chess.WHITE:
                return PIECE_VALUES[chess.KING]
            elif winner == chess.BLACK:
                return -PIECE_VALUES[chess.KING]
            else:
                return 0

        output_vector = self.tensor_eval(state)
        return output_vector.item()


bnet = BasicNetwork()
bnet.evaluate(chess.Board())
