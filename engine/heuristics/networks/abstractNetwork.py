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
    def board_to_tensor(state, data_type=torch.int8):
        board_tensor = torch.zeros((11, 64), dtype=data_type)
        piece_types = [
            state.pawns,
            state.knights,
            state.bishops,
            state.rooks,
            state.queens,
            state.kings,
        ]
        white_locations = state.occupied_co[chess.WHITE]
        black_locations = state.occupied_co[chess.BLACK]

        for layer, bitboard in enumerate(piece_types):
            white_bb_squares = chess.SquareSet(bitboard & white_locations)
            if white_bb_squares:
                board_tensor[layer, list(white_bb_squares)] = 1

            black_bb_squares = chess.SquareSet(bitboard & black_locations)
            if black_bb_squares:
                board_tensor[layer, list(black_bb_squares)] = 1

        if state.has_kingside_castling_rights(chess.WHITE):
            board_tensor[6, :] = 1
        if state.has_queenside_castling_rights(chess.WHITE):
            board_tensor[7, :] = 1
        if state.has_kingside_castling_rights(chess.BLACK):
            board_tensor[8, :] = 1
        if state.has_queenside_castling_rights(chess.BLACK):
            board_tensor[9, :] = 1
        if state.turn:
            board_tensor[10, :] = 1

        return board_tensor

    def tensor_eval(self, state):
        input_vector = self.board_to_tensor(state, data_type=torch.float32)
        with torch.no_grad():
            output_vector = self.model(input_vector)
        return output_vector

    def evaluate(self, state):
        outcome = state.outcome()
        if outcome is not None:
            winner = outcome.winner
            return OUTCOMES[winner]

        output_vector = self.tensor_eval(state)
        return output_vector.item()
