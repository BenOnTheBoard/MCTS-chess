from bulletchess import (
    BLACK,
    WHITE,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    SQUARES,
    BLACK_KINGSIDE,
    BLACK_QUEENSIDE,
    WHITE_KINGSIDE,
    WHITE_QUEENSIDE,
)
import torch

from engine.heuristics.heuristicInterface import HeuristicInterface


class AbstractNetwork(HeuristicInterface):
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.init_model()

    def init_model(self):
        raise NotImplementedError("Model forms must be supplied by subclass.")

    @staticmethod
    def board_to_tensor(state, data_type=torch.int8):
        board_tensor = torch.zeros((11, 64), dtype=data_type)
        piece_types = (
            state[PAWN],
            state[KNIGHT],
            state[BISHOP],
            state[ROOK],
            state[QUEEN],
            state[KING],
        )
        white_mask = state[WHITE]
        black_mask = state[BLACK]

        for layer, bitboard in enumerate(piece_types):
            white_idxs = [SQUARES.index(sq) for sq in (bitboard & white_mask)]
            black_idxs = [SQUARES.index(sq) for sq in (bitboard & black_mask)]
            if white_idxs:
                board_tensor[layer, white_idxs] = 1
            if black_idxs:
                board_tensor[layer, black_idxs] = -1

        rights = state.castling_rights
        if WHITE_KINGSIDE in rights:
            board_tensor[6, :] = 1
        if WHITE_QUEENSIDE in rights:
            board_tensor[7, :] = 1
        if BLACK_KINGSIDE in rights:
            board_tensor[8, :] = 1
        if BLACK_QUEENSIDE in rights:
            board_tensor[9, :] = 1
        if state.turn is WHITE:
            board_tensor[10, :] = 1

        return board_tensor

    def tensor_eval(self, state):
        input_vector = self.board_to_tensor(state, data_type=torch.float32)
        with torch.no_grad():
            output_vector = self.model(input_vector)
        return output_vector

    def evaluate(self, state):
        output_vector = self.tensor_eval(state)
        return output_vector.item()
