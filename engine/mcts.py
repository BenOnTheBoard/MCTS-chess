from copy import deepcopy
from time import perf_counter

from engine.node import Node
from engine.utils import get_best_move


class MCTS:
    def __init__(self, position, time_out, tree_evaluator, rollout_heuristic):
        self.time_out = time_out  # sec
        self.tree_evaluator = tree_evaluator
        self.rollout_heuristic = rollout_heuristic

        self.set_position(position)

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

    def tree_policy(self, node, is_white):
        if node.is_leaf():
            return node

        if is_white:
            best_value = -float("inf")
        else:
            best_value = float("inf")
        best_node = None

        for child in node.children:
            if child.visits == 0:
                return child

            child_value = self.tree_evaluator.evaluate(child, node, is_white)

            if is_white:
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

            result = self.rollout_heuristic.evaluate(state)

            while node.has_parent():
                node.update(result)
                node = node.parent
            self.root_node.update(result)

        return get_best_move(self.root_node, self.position.turn)
