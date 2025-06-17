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

        # all slides
        for dir_idx, (dr, dc) in enumerate(DIRECTIONS):
            for dist in range(1, 8):
                if row_diff == dr * dist and col_diff == dc * dist:
                    return dir_idx * 7 + (dist - 1)

        for idx, (dr, dc) in enumerate(KNIGHTS_MOVES):
            if row_diff == dr and col_diff == dc:
                return 56 + idx

        # underpromotions, 3 types × 3 directions = 9
        if promotion in AbstractPolicyNetwork.UNDERPROMOTION_TYPES:
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
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion
        plane = AbstractPolicyNetwork.move_to_plane(from_sq, to_sq, promotion)
        if plane is not None:
            row, col = divmod(from_sq, 8)
            tensor[plane, row, col] = 1.0
        return tensor

    def board_to_legal_moves_mask(self, board):
        tensor = torch.zeros((73, 8, 8), dtype=torch.float32)
        for move in board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            promotion = move.promotion
            plane = AbstractPolicyNetwork.move_to_plane(from_sq, to_sq, promotion)
            if plane is not None:
                row, col = divmod(from_sq, 8)
                tensor[plane, row, col] = 1.0
        return tensor

    def get_move_distribution(self, state):
        distribution = self.tensor_eval(state)
        mask = self.board_to_legal_moves_mask(state)

        return distribution * mask
