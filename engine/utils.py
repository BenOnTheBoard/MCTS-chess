import chess
import math

from engine.values import PIECE_VALUES


def captured_piece(state, move):
    if state.is_en_passant(move):
        return chess.PAWN
    else:
        return state.piece_at(move.to_square).piece_type


def material_balance(piece_map):
    """
    Takes a piece map and returns a value between 0 and 1,
    which describes which side the position favours.
    """
    diff = 0
    for piece in piece_map.values():
        if piece.color:
            diff += PIECE_VALUES[piece.piece_type]
        else:
            diff -= PIECE_VALUES[piece.piece_type]

    return 1 / (1 + math.exp(-diff / 200))


def node_comparator(node):
    if node.visits == 0:
        return 0.5
    return node.quality


def get_best_move(node, is_white):
    node.children.sort(key=node_comparator)
    if is_white:
        return node.children[-1].move
    else:
        return node.children[0].move
