import random

from engine.heuristics.rollouts.perMoveHeuristic import PerMoveHeuristic


class Random(PerMoveHeuristic):
    def select_move(self, state):
        move_list = list(state.legal_moves)
        return random.choice(move_list)
