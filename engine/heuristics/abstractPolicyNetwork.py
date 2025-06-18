import chess
import torch

from engine.heuristics.abstractNetwork import AbstractNetwork
from engine.values import DIRECTIONS, KNIGHTS_MOVES


class AbstractPolicyNetwork(AbstractNetwork):
    UNDERPROMOTION_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

    @staticmethod
    def move_to_plane(from_square, to_square, promotion=None):
        from_row, from_col = divmod(from_square, 8)
        to_row, to_col = divmod(to_square, 8)
        row_diff = to_row - from_row
        col_diff = to_col - from_col

        if promotion not in AbstractPolicyNetwork.UNDERPROMOTION_TYPES:
            # all slides, inc queen promotions
            for dir_idx, (dr, dc) in enumerate(DIRECTIONS):
                if (dr > 0) - (dr < 0) != (row_diff > 0) - (row_diff < 0):
                    continue
                if (dc > 0) - (dc < 0) != (col_diff > 0) - (col_diff < 0):
                    continue

                if dr == 0:
                    dist = col_diff // dc
                    return dir_idx * 7 + (dist - 1)

                dist = row_diff // dr
                if dc == 0 or dist == col_diff // dc:
                    return dir_idx * 7 + (dist - 1)

            try:
                idx = KNIGHTS_MOVES.index((row_diff, col_diff))
                return 56 + idx
            except ValueError:
                pass

        # underpromotions, 3 types × 3 directions = 9
        if abs(row_diff) == 1 or (row_diff == 0 and abs(col_diff) == 1):
            for prom_idx, prom_piece in enumerate(
                AbstractPolicyNetwork.UNDERPROMOTION_TYPES
            ):
                if promotion == prom_piece:
                    # Straight
                    if col_diff == 0:
                        return 64 + prom_idx * 3
                    # Left capture
                    elif col_diff == -1:
                        return 64 + prom_idx * 3 + 1
                    # Right capture
                    elif col_diff == 1:
                        return 64 + prom_idx * 3 + 2

        return None

    @staticmethod
    def move_to_tensor(move):
        tensor = torch.zeros((73, 8, 8), dtype=torch.float32)
        plane = AbstractPolicyNetwork.move_to_plane(
            move.from_square, move.to_square, move.promotion
        )
        if plane is not None:
            row, col = divmod(move.from_square, 8)
            tensor[plane, row, col] = 1.0
        return tensor

    @staticmethod
    def board_to_legal_moves_mask(board):
        tensor = torch.zeros((73, 8, 8), dtype=torch.float32)
        for move in board.legal_moves:
            plane = AbstractPolicyNetwork.move_to_plane(
                move.from_square, move.to_square, move.promotion
            )
            if plane is not None:
                row, col = divmod(move.from_square, 8)
                tensor[plane, row, col] = 1.0
        return tensor

    def get_masked_move_distribution(self, state):
        distribution = self.tensor_eval(state)
        mask = self.board_to_legal_moves_mask(state)

        return distribution * mask
