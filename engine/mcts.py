from tqdm import tqdm

from engine.node import Node
from engine.utils import get_best_move
from engine.LRUCache import LRUCache


class MCTS:
    def __init__(
        self,
        position,
        tree_evaluator,
        rollout_heuristic,
        backpropagation_rule,
    ):
        self.tree_evaluator = tree_evaluator
        self.rollout_heuristic = rollout_heuristic
        self.backpropagation_rule = backpropagation_rule
        self.LRUCache = LRUCache(maxsize=100_000)

        self.set_position(position)

    def set_position(self, new_position):
        self.root_node = Node(None, new_position.turn, None)
        self.position = new_position.copy()

    def add_move(self, move):
        found_child = False
        if self.root_node.children is not None:
            for child in self.root_node.children:
                if child.move == move:
                    child.parent = None
                    self.root_node = child
                    found_child = True
                    break
        if not found_child:
            self.root_node = Node(None, not self.root_node.turn, None)

        self.position.push(move)

    def tree_policy(self, node):
        if node.is_leaf():
            return node

        if node.turn:
            best_value = -float("inf")
        else:
            best_value = float("inf")
        best_node = None

        for child in node.children:
            if child.visits == 0:
                return child

            child_value = self.tree_evaluator.evaluate(child, node)

            if node.turn:
                if child_value > best_value:
                    best_value = child_value
                    best_node = child
            else:
                if child_value < best_value:
                    best_value = child_value
                    best_node = child

        return best_node

    def evaluate_state(self, state):
        result = self.LRUCache.get(state)
        if result is None:
            result = self.rollout_heuristic.evaluate(state)
            self.LRUCache.put(state, result)
        return result

    def propagate_updates(self, node, value):
        while node.has_parent():
            new_quality, next_value = self.backpropagation_rule.calculate(node, value)
            node.update_quality(new_quality)
            value = next_value
            node = node.parent
        new_quality, _ = self.backpropagation_rule.calculate(node, value)
        node.update_quality(new_quality)

    def get_move(self, node_count):
        for _ in tqdm(range(node_count)):
            node, state = self.root_node, self.position.copy(stack=False)

            while not node.is_leaf():
                node = self.tree_policy(node)
                state.push(node.move)

            node.expand_node(state)

            node = self.tree_policy(node)
            state.push(node.move)

            result = self.evaluate_state(state)
            self.propagate_updates(node, result)

        return get_best_move(self.root_node)
