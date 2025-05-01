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
        bit_vector = torch.tensor(bit_list, dtype=torch.int8)
        return bit_vector

    @staticmethod
    def board_to_tensor(state):
        input_sections = []
        for piece in chess.PIECE_TYPES:
            white_int = state.pieces_mask(piece, chess.WHITE)
            white_section = AbstractNetwork.int_to_bit_vector(white_int)

            black_int = state.pieces_mask(piece, chess.BLACK)
            black_section = AbstractNetwork.int_to_bit_vector(black_int)

            piece_section = white_section - black_section

            input_sections.append(piece_section)

        return torch.concatenate(input_sections)

    def tensor_eval(self, state):
        input_vector = self.board_to_tensor(state).to(torch.float32)
        output_vector = self.model(input_vector)
        return output_vector

    def evaluate(self, state):
        outcome = state.outcome(claim_draw=True)
        if outcome is not None:
            winner = outcome.winner
            return OUTCOMES[winner]

        output_vector = self.tensor_eval(state)
        return output_vector.item()


if __name__ == "__main__":
    AbstractNetwork.board_to_tensor(chess.Board())
