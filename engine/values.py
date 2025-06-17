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

DIRECTIONS = (
    (-1, 0),  # S, e4 is 2N
    (-1, 1),
    (0, 1),  # E
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)

KNIGHTS_MOVES = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)
