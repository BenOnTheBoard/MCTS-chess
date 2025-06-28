from bulletchess import QUEEN, SQUARES, PIECE_TYPES
import torch
from torch.nn.functional import softmax

from engine.heuristics.abstractNetwork import AbstractNetwork
from engine.values import DIRECTIONS, KNIGHTS_MOVES


class AbstractPolicyNetwork(AbstractNetwork):
    @staticmethod
    def move_to_plane(from_square, to_square, promotion_type=None):
        from_row, from_col = divmod(SQUARES.index(from_square), 8)
        to_row, to_col = divmod(SQUARES.index(to_square), 8)
        row_diff = to_row - from_row
        col_diff = to_col - from_col

        if promotion_type is None or promotion_type == QUEEN:
            row_diff_sgn = (row_diff > 0) - (row_diff < 0)
            col_diff_sgn = (col_diff > 0) - (col_diff < 0)

            try:
                dir_idx = DIRECTIONS.index((row_diff_sgn, col_diff_sgn))
                if row_diff_sgn == 0:
                    dist = col_diff // col_diff_sgn
                    return dir_idx * 7 + (dist - 1)

                dist = row_diff // row_diff_sgn
                if col_diff_sgn == 0 or dist == col_diff // col_diff_sgn:
                    return dir_idx * 7 + (dist - 1)
            except ValueError:
                pass

            try:
                idx = KNIGHTS_MOVES.index((row_diff, col_diff))
                return 56 + idx
            except ValueError:
                pass

        # col_diff is -1,0,1, left, straight, right
        # underpromotion is 1,2,3, knight, bishop, rook
        promotion = PIECE_TYPES.index(promotion_type)
        return 64 + (promotion - 1) * 3 + 1 + col_diff

    @staticmethod
    def move_to_flat_index(move):
        plane = AbstractPolicyNetwork.move_to_plane(
            move.origin, move.destination, move.promotion
        )
        row, col = divmod(SQUARES.index(move.origin), 8)
        return plane * 64 + row * 8 + col

    @staticmethod
    def board_to_legal_moves_mask(board):
        tensor = torch.zeros((73, 8, 8), dtype=torch.float32)
        for move in board.legal_moves:
            plane = AbstractPolicyNetwork.move_to_plane(
                move.from_square, move.to_square, move.promotion
            )
            row, col = divmod(move.from_square, 8)
            tensor[plane, row, col] = 1.0
        return tensor

    def get_masked_move_distribution(self, state):
        distribution = self.tensor_eval(state)
        dist_shape = distribution.shape
        dist_soft = softmax(distribution.view(-1), dim=0).view(dist_shape)

        mask = self.board_to_legal_moves_mask(state)

        return dist_soft * mask
