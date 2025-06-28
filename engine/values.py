import bulletchess

PIECE_VALUES = {
    bulletchess.PAWN: 100,
    bulletchess.KNIGHT: 305,
    bulletchess.BISHOP: 333,
    bulletchess.ROOK: 563,
    bulletchess.QUEEN: 950,
    bulletchess.KING: 10_000,
}

OUTCOMES = {
    bulletchess.WHITE: 1,
    bulletchess.BLACK: -1,
    None: 0,
}

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
