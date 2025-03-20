from engine.heuristics.heuristicInterface import HeuristicInterface

from engine.utils import material_balance
from engine.values import OUTCOMES


class PerMoveHeuristic(HeuristicInterface):
    def __init__(self):
        super().__init__()

    def evaluate(self, state):
        state_piece_count = len(state.piece_map())
        while not state.is_game_over():
            p_map = state.piece_map()
            if len(p_map) <= 6 and state_piece_count > 8:
                return material_balance(p_map)

            choice_move = self.select_move(state)
            state.push(choice_move)

        return OUTCOMES[state.outcome().winner]

    def select_move(self, state):
        """
        Return a move based on a heuristic
        """
        raise NotImplementedError()
