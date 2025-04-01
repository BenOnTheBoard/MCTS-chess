import chess
import torch
import torch.nn as nn

from engine.heuristics.heuristicInterface import HeuristicInterface
from engine.values import PIECE_VALUES


class ConvNetwork(HeuristicInterface):
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = torch.nn.Sequential(
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
    def int_to_bit_vector(num):
        bit_list = [int(bit) for bit in bin(num)[2:].zfill(64)]
        bit_vector = torch.tensor(bit_list, dtype=torch.float)
        return bit_vector

    @staticmethod
    def board_to_tensor(state):
        input_sections = []
        for colour in chess.COLORS:
            for piece in chess.PIECE_TYPES:
                pc_int = state.pieces_mask(piece, colour)
                section = ConvNetwork.int_to_bit_vector(pc_int)
                input_sections.append(section)

        return torch.concatenate(input_sections)

    def tensor_eval(self, state):
        input_vector = self.board_to_tensor(state)
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
