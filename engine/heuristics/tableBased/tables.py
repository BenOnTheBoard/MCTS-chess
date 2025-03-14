import chess
import numpy as np

TABLES = {
    chess.PAWN: np.zeros((8, 8)),
    chess.KNIGHT: np.zeros((8, 8)),
    chess.BISHOP: np.zeros((8, 8)),
    chess.ROOK: np.zeros((8, 8)),
    chess.QUEEN: np.zeros((8, 8)),
    chess.KING: np.zeros((8, 8)),
}
