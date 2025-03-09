import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 305,
    chess.BISHOP: 333,
    chess.ROOK: 563,
    chess.QUEEN: 950,
    chess.KING: 10_000,
}

OUTCOMES = {chess.WHITE: 1, chess.BLACK: 0, None: 0.5}
