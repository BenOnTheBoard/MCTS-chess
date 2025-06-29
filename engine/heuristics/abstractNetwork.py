import bulletchess
from bulletchess import (
    CHECKMATE,
    DRAW,
    BLACK,
    WHITE,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    SQUARES,
)
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
    def board_to_tensor(state, data_type=torch.int8):
        board_tensor = torch.zeros((11, 64), dtype=data_type)
        piece_types = [
            state[PAWN],
            state[KNIGHT],
            state[BISHOP],
            state[ROOK],
            state[QUEEN],
            state[KING],
        ]
        white_locations = state[WHITE]
        black_locations = state[BLACK]

        for layer, bitboard in enumerate(piece_types):
            white_bb_squares = bitboard & white_locations
            if white_bb_squares:
                white_ints = [SQUARES.index(sq) for sq in white_bb_squares]
                board_tensor[layer, white_ints] = 1

            black_bb_squares = bitboard & black_locations
            if black_bb_squares:
                black_ints = [SQUARES.index(sq) for sq in black_bb_squares]
                board_tensor[layer, black_ints] = -1

        if state.castling_rights.kingside(bulletchess.WHITE):
            board_tensor[6, :] = 1
        if state.castling_rights.queenside(bulletchess.WHITE):
            board_tensor[7, :] = 1
        if state.castling_rights.kingside(bulletchess.BLACK):
            board_tensor[8, :] = 1
        if state.castling_rights.queenside(bulletchess.BLACK):
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
        if state in CHECKMATE:
            return -OUTCOMES[state.turn], None
        if state in DRAW:
            return OUTCOMES[None], None

        output_vector = self.tensor_eval(state)
        return output_vector.item()
