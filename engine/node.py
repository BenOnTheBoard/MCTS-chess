from chess.polyglot import zobrist_hash
from copy import copy


class Node:
    hash_table = dict()

    def __init__(self, move, parent):
        self.parent_move_dict = {parent: move}
        self.children = None
        self.score = 0
        self.visits = 0

    def expand_node(self, state):
        if self.children is None and not state.is_game_over():
            self.children = []
            new_state = copy(state)
            for move in state.legal_moves:
                new_state.push(move)
                new_zh = zobrist_hash(new_state)

                if new_zh in Node.hash_table:
                    new_child = Node.hash_table[new_zh]
                    new_child.parent_move_dict[self] = move
                else:
                    new_child = Node(move, self)
                    Node.hash_table[new_zh] = new_child

                self.children.append(new_child)

    def update(self, result):
        self.visits += 1
        self.score += result

    def is_leaf(self):
        return self.children is None

    def has_parent(self):
        return self.parent_move_dict is not None

    def get_move(self, parent):
        return self.parent_move_dict[parent]

    def get_parents(self):
        return self.parent_move_dict.keys()
