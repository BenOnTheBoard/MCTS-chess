import chess
import torch

from engine.heuristics.heuristicInterface import HeuristicInterface
from engine.values import OUTCOMES


class AbstractNetwork(HeuristicInterface):
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.init_model()

    def init_model(self):
        raise NotImplementedError("Model forms must be supplied by subclass.")

    @staticmethod
    def int_to_bit_vector(num):
        bit_list = [int(bit) for bit in bin(num)[2:].zfill(64)]
        bit_vector = torch.tensor(bit_list, dtype=torch.bool)
        return bit_vector

    @staticmethod
    def board_to_tensor(state):
        input_sections = []
        for colour in chess.COLORS:
            for piece in chess.PIECE_TYPES:
                pc_int = state.pieces_mask(piece, colour)
                section = AbstractNetwork.int_to_bit_vector(pc_int)
                input_sections.append(section)

        return torch.concatenate(input_sections)

    def tensor_eval(self, state):
        input_vector = self.board_to_tensor(state)
        output_vector = self.model(input_vector)
        return output_vector

    def evaluate(self, state):
        if state.is_game_over():
            winner = state.outcome().winner
            return OUTCOMES[winner]

        output_vector = self.tensor_eval(state)
        return output_vector.item()
