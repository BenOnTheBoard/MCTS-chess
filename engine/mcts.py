import math
from copy import deepcopy
from time import perf_counter

import chess

from engine.heuristics.bestCapture import BestCapture
from engine.node import Node
from engine.utils import (
    get_best_move,
    material_balance,
    node_comparator,
)
from engine.values import OUTCOMES


class MCTS:
    def __init__(self, position):
        self.root_node = Node(None, None)
        self.count = 0
        self.time_out = 120  # sec
        self.position = position

        self.seen_states = {}

        self.rollout_heuristic = BestCapture()

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
        state_str = " ".join(state.fen().split()[:-2])
        if state_str in self.seen_states:
            return self.seen_states[state_str]

        while not state.is_game_over():
            p_map = state.piece_map()
            if len(p_map) <= 6 and state_piece_count > 8:
                mat_bal = material_balance(p_map)
                self.seen_states[state_str] = mat_bal
                return mat_bal

            choice_move = self.rollout_heuristic.evaluate(state)
            state.push(choice_move)

        result = OUTCOMES[state.outcome().winner]
        self.seen_states[state_str] = result
        return result

    def get_move(self):
        start = perf_counter()
        count = 0
        while (perf_counter() - start) < self.time_out:
            count += 1
            if count % 1000 == 0:
                elapsed_time = perf_counter() - start
                print(f"""
                Elapsed time: {elapsed_time // 60} min and {elapsed_time % 60:.2f} s
                Iters: {count} it
                Rate: {(count / elapsed_time):.2f} it/s
                """)

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

        print(f"Runs: {count}")
        self.root_node.children.sort(key=node_comparator)
        for top_kid in self.root_node.children:
            print(top_kid.move, node_comparator(top_kid))

        return get_best_move(self.root_node, self.position.turn)


# start_fen = "rnb1kb1r/ppp1pppp/5n2/3q4/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1"
# start_fen = "rnbqkb1r/ppp1pppp/8/8/2n5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1"
start_fen = "5k1n/5ppp/7N/2q5/8/8/1K6/6R1 w - - 2 2"
origin = chess.Board(fen=start_fen)
print(origin)
mcts = MCTS(origin)
print("Chose:", mcts.get_move())
