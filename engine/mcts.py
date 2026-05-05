from bulletchess import CHECKMATE, DRAW
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

        self.indexer = self.network.move_to_flat_index
        self.evaluator = self.network.evaluate

    def set_position(self, new_position):
        self.root_node = Node(None, None)
        self.position = new_position.copy()

    def add_move(self, move):
        self.position.apply(move)
        if self.root_node.children is not None:
            for child in self.root_node.children:
                if child.move == move:
                    self.root_node = child
                    return
        self.root_node = Node(None, None)

    def PUCT(self, node, turn):
        sqrtv = sqrt(node.visits)
        sign = OUTCOMES[turn]

        def PUCT_node(child):
            exploring_term = sqrtv / (1 + child.visits)
            delta = self.exploration * child.prior * exploring_term
            return sign * child.quality + delta

        return PUCT_node

    def tree_policy(self, node, turn):
        for child in node.children:
            if child.visits == 0:
                return child

        evaluator = self.PUCT(node, turn)
        return max(node.children, key=evaluator)

    def expand_node(self, node, state, move_dist):
        legal_moves = tuple(state.legal_moves())
        flat_dist = move_dist.flatten()
        idxs = [self.indexer(move) for move in legal_moves]
        probs = softmax(flat_dist[idxs], dim=0).tolist()
        node.children = tuple(Node(m, p) for p, m in zip(probs, legal_moves))

    def evaluate_state(self, state):
        cached_pair = self.LRUCache.get(state)
        if cached_pair is not None:
            return cached_pair

        eval_pair = self.evaluator(state)
        self.LRUCache.put(state, eval_pair)
        return eval_pair

    def get_move(self, node_count, tqdm_on=False):
        if tqdm_on:
            counter = tqdm(range(node_count))
        else:
            counter = range(node_count)

        for _ in counter:
            node, state = self.root_node, self.position.copy()
            path = [node]
            while node.children is not None:
                node = self.tree_policy(node, state.turn)
                state.apply(node.move)
                path.append(node)

            if state in CHECKMATE:
                result = -OUTCOMES[state.turn]
            elif state in DRAW:
                result = OUTCOMES[None]
            else:
                result, move_distribution = self.evaluate_state(state)
                self.expand_node(node, state, move_distribution)

            for path_node in path:
                path_node.update_quality(result)

        most_visited = max(self.root_node.children, key=lambda n: n.visits)
        return most_visited.move
