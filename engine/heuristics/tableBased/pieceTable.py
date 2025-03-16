from math import exp

import chess

from engine.heuristics.heuristicInterface import HeuristicInterface
from engine.heuristics.tableBased.tables import TABLES
from engine.values import PIECE_VALUES, OUTCOMES


class PieceTable(HeuristicInterface):
    def __init__(self, tables=TABLES):
        self.tables = tables

    def evaluate(self, state):
        if state.is_game_over():
            winner = state.outcome().winner
            if winner == chess.WHITE:
                return PIECE_VALUES[chess.KING]
            elif winner == chess.BLACK:
                return -PIECE_VALUES[chess.KING]
            else:
                return 0

        score = 0
        for square in chess.SQUARES:
            row, col = divmod(square, 8)
            piece = state.piece_at(square)
            if piece is not None:
                piece_val = PIECE_VALUES[piece.piece_type]
                piece_table = self.tables[piece.piece_type]
                if piece.color == chess.WHITE:
                    piece_bonus = piece_table[7 - row][col]
                    score += piece_val + piece_bonus
                else:
                    piece_bonus = piece_table[row][col]
                    score -= piece_val + piece_bonus

        return score
