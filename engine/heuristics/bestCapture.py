import random

from engine.heuristics.heuristicInterface import HeuristicInterface
from engine.utils import captured_piece


class BestCapture(HeuristicInterface):
    def evaluate(self, state):
        move_list = []
        best_capture = None
        best_captured_piece = -1
        for move in state.legal_moves:
            move_list.append(move)
            if state.is_capture(move):
                captured = captured_piece(state, move)
                if captured > best_captured_piece:
                    best_capture = move
                    best_captured_piece = captured

        if best_capture is not None:
            return best_capture
        else:
            return random.choice(move_list)
