import random

from engine.heuristics.heuristicInterface import HeuristicInterface


class Random(HeuristicInterface):
    def evaluate(self, state):
        move_list = list(state.legal_moves)
        return random.choice(move_list)
