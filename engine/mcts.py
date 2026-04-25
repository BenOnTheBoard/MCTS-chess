from bulletchess import CHECKMATE, DRAW, WHITE
from math import sqrt
from torch.nn.functional import softmax
from tqdm import tqdm

from engine.node import Node
from engine.LRUCache import LRUCache
from engine.values import OUTCOMES


class MCTS:
    def __init__(self, position, network, exploration):
        self.network = network
        self.LRUCache = LRUCache(maxsize=50_000)
        self.exploration = exploration
        self.set_position(position)

    def set_position(self, new_position):
        self.root_node = Node(None, new_position.turn, None, None)
        self.position = new_position.copy()

    def add_move(self, move):
        self.position.apply(move)
        if self.root_node.children is not None:
            for child in self.root_node.children:
                if child.move == move:
                    child.parent = None
                    self.root_node = child
                    return
        self.root_node = Node(None, ~self.root_node.turn, None, None)

    def PUCT(self, child, node):
        exploring_term = sqrt(node.visits) / (1 + child.visits)
        delta = self.exploration * child.prior * exploring_term

        if node.turn is WHITE:
            return child.quality + delta
        else:
            return child.quality - delta

    def tree_policy(self, node):
        for child in node.children:
            if child.visits == 0:
                return child

        if node.turn is WHITE:
            return max(node.children, key=lambda c: self.PUCT(c, node))
        return min(node.children, key=lambda c: self.PUCT(c, node))

    def expand_node(self, node, state, move_distribution):
        flat_dist = move_distribution.flatten()
        idxs = [self.network.move_to_flat_index(move) for move in state.legal_moves()]
        probs = softmax(flat_dist[idxs], dim=0)
        node.children = tuple(
            Node(move, ~node.turn, probs[i].item(), node)
            for i, move in enumerate(state.legal_moves())
        )

    def evaluate_state(self, state):
        cached_pair = self.LRUCache.get(state)
        if cached_pair is not None:
            return cached_pair

        eval_pair = self.network.evaluate(state)
        self.LRUCache.put(state, eval_pair)
        return eval_pair

    def propagate_updates(self, node, value):
        while True:
            new_quality = node.quality + (value - node.quality) / (node.visits + 1)
            node.update_quality(new_quality)
            node = node.parent
            if node is None:
                return

    def get_move(self, node_count, tqdm_on=False):
        if tqdm_on:
            counter = tqdm(range(node_count))
        else:
            counter = range(node_count)

        for _ in counter:
            node, state = self.root_node, self.position.copy()

            while not node.is_leaf():
                node = self.tree_policy(node)
                state.apply(node.move)

            if state in CHECKMATE:
                result = -OUTCOMES[state.turn]
            elif state in DRAW:
                result = OUTCOMES[None]
            else:
                result, move_distribution = self.evaluate_state(state)
                self.expand_node(node, state, move_distribution)

            self.propagate_updates(node, result)

        most_visited = max(self.root_node.children, key=lambda n: n.visits)
        return most_visited.move
