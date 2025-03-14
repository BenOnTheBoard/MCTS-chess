from math import exp

import chess

from engine.heuristics.heuristicInterface import HeuristicInterface
from engine.heuristics.tableBased.tables import TABLES
from engine.values import PIECE_VALUES


class PieceTable(HeuristicInterface):
    def evaluate(self, state):
        score = 0

        for square in chess.SQUARES:
            row, col = divmod(square, 8)
            piece = state.piece_at(square)
            if piece is not None:
                piece_val = PIECE_VALUES[piece.piece_type]
                piece_modifier = TABLES[piece.piece_type][row][col]

                if piece.color == chess.WHITE:
                    score += piece_val + piece_modifier
                else:
                    score -= piece_val + piece_modifier

        return 1 / (1 + exp(-score / 200))
