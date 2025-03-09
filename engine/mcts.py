import math
from copy import deepcopy
from time import perf_counter

from engine.heuristics.bestCapture import BestCapture
from engine.node import Node
from engine.utils import get_best_move, material_balance
from engine.values import OUTCOMES


class MCTS:
    def __init__(self, position, time_out):
        self.root_node = Node(None, None)
        self.time_out = time_out  # sec
        self.position = position.copy()

        self.rollout_heuristic = BestCapture()

    def add_move(self, move):
        found_child = False
        for child in self.root_node.children:
            if child.move == move:
                child.parent = None
                self.root_node = child
                found_child = True
                break
        if not found_child:
            print("Not really meant to happen.")
            self.root_node = Node(None, None)

        self.position.push(move)

    def tree_policy(self, node, is_maximizing_player):
        if node.is_leaf():
            return node

        if is_maximizing_player:
            best_uct = -float("inf")
        else:
            best_uct = float("inf")
        best_node = None

        for child in node.children:
            if child.visits == 0:
                return child

            child_quality = child.score / child.visits
            exploring_term = math.sqrt(2 * math.log(node.visits) / child.visits)

            if is_maximizing_player:
                uct_value = child_quality + exploring_term
                if uct_value > best_uct:
                    best_uct = uct_value
                    best_node = child
            else:
                uct_value = child_quality - exploring_term
                if uct_value < best_uct:
                    best_uct = uct_value
                    best_node = child

        return best_node

    def rollout_policy(self, state):
        state_piece_count = len(state.piece_map())
        while not state.is_game_over():
            p_map = state.piece_map()
            if len(p_map) <= 6 and state_piece_count > 8:
                return material_balance(p_map)

            choice_move = self.rollout_heuristic.evaluate(state)
            state.push(choice_move)

        return OUTCOMES[state.outcome().winner]

    def get_move(self):
        start = perf_counter()
        while (perf_counter() - start) < self.time_out:
            node, state = self.root_node, deepcopy(self.position)

            while not node.is_leaf():
                node = self.tree_policy(node, state.turn)
                state.push(node.move)

            node.expand_node(state)
            node = self.tree_policy(node, state.turn)

            result = self.rollout_policy(state)

            while node.has_parent():
                node.update(result)
                node = node.parent
            self.root_node.update(result)

        return get_best_move(self.root_node, self.position.turn)


# start_fen = "rnbqkb1r/ppp1pppp/8/8/2n5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1"
# start_fen = "6kn/3q1ppp/8/8/6N1/8/1K6/6R1 w - - 0 1"
# start_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
# origin = chess.Board(fen=start_fen)
# print(origin)
# mcts = MCTS(origin, 10)
# print("Chose:", mcts.get_move())
