import math
import random
from copy import deepcopy
from time import perf_counter

import chess

from node import Node
from utils import (
    captured_piece,
    get_best_move,
    material_balance,
    node_comparator,
)


class MCTS:
    def __init__(self, position):
        self.root_node = Node(None, None)
        self.count = 0
        self.time_out = 120  # sec
        self.position = position

    def tree_policy_child(self, node, is_maximizing_player):
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

            child_ratio = child.score / child.visits

            if is_maximizing_player:
                uct_value = child_ratio + math.sqrt(
                    2 * math.log(node.visits) / child.visits
                )
                if uct_value > best_uct:
                    best_uct = uct_value
                    best_node = child
            else:
                uct_value = child_ratio - math.sqrt(
                    2 * math.log(node.visits) / child.visits
                )
                if uct_value < best_uct:
                    best_uct = uct_value
                    best_node = child

        return best_node

    def simulation_policy_child(self, state):
        while not state.is_game_over():
            p_map = state.piece_map()
            if len(p_map) <= 6:
                return material_balance(p_map)

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
                choice_move = best_capture
            else:
                choice_move = random.choice(move_list)
            state.push(choice_move)

        outcome = state.outcome().winner
        if outcome == chess.WHITE:
            return 1
        elif outcome == chess.BLACK:
            return 0
        else:
            return 0.5

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

            while not node.is_leaf():  # select leaf
                node = self.tree_policy_child(node, state.turn)
                state.push(node.move)

            node.expand_node(state)  # expand
            node = self.tree_policy_child(node, state.turn)

            result = self.simulation_policy_child(state)  # simulate

            while node.has_parent():  # propagate
                node.update(result)
                node = node.parent
            self.root_node.update(result)

        print(f"Runs: {count}")
        self.root_node.children.sort(key=node_comparator)
        for top_kid in self.root_node.children:
            print(top_kid.move, node_comparator(top_kid))

        return get_best_move(self.root_node, self.position.turn)


# start_fen = "rnb1kb1r/ppp1pppp/5n2/3q4/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1"
start_fen = "rnbqkb1r/ppp1pppp/8/8/2n5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1"
origin = chess.Board(fen=start_fen)
print(origin)
mcts = MCTS(origin)
print("Chose:", mcts.get_move())
