from copy import deepcopy
from time import perf_counter

from engine.heuristics.bestCapture import BestCapture
from engine.node import Node
from engine.nodeEvaluators.UCT import UCT
from engine.utils import get_best_move, material_balance
from engine.values import OUTCOMES


class MCTS:
    def __init__(self, position, time_out):
        self.root_node = Node(None, None)
        self.time_out = time_out  # sec
        self.position = position.copy()

        self.node_evaluator = UCT(1.4)
        self.rollout_heuristic = BestCapture()

    def set_position(self, new_position):
        self.root_node = Node(None, None)
        self.position = new_position.copy()

    def add_move(self, move):
        found_child = False
        for child in self.root_node.children:
            if child.move == move:
                child.parent = None
                self.root_node = child
                found_child = True
                break
        if not found_child:
            self.root_node = Node(None, None)

        self.position.push(move)

    def tree_policy(self, node, is_max_player):
        if node.is_leaf():
            return node

        if is_max_player:
            best_value = -float("inf")
        else:
            best_value = float("inf")
        best_node = None

        for child in node.children:
            if child.visits == 0:
                return child

            child_value = self.node_evaluator.evaluate(child, node, is_max_player)

            if is_max_player:
                if child_value > best_value:
                    best_value = child_value
                    best_node = child
            else:
                if child_value < best_value:
                    best_value = child_value
                    best_node = child

        return best_node

    def get_move(self):
        start = perf_counter()
        while (perf_counter() - start) < self.time_out:
            node, state = self.root_node, deepcopy(self.position)

            while not node.is_leaf():
                node = self.tree_policy(node, state.turn)
                state.push(node.move)

            node.expand_node(state)
            node = self.tree_policy(node, state.turn)

            result = self.rollout_heuristic.evaluate(state)

            while node.has_parent():
                node.update(result)
                node = node.parent
            self.root_node.update(result)

        return get_best_move(self.root_node, self.position.turn)
