from chess.polyglot import zobrist_hash
from copy import copy


class Node:
    hash_table = dict()

    def __init__(self, move, parent):
        if move is None:
            self.parent_move_dict = {}
        else:
            self.parent_move_dict = {parent: move}

        self.children = None
        self.score = 0
        self.visits = 0

    def expand_node(self, state):
        if self.children is not None or state.is_game_over():
            return

        self.children = []
        for move in state.legal_moves:
            state.push(move)
            zh = zobrist_hash(state)

            if zh in Node.hash_table:
                child = Node.hash_table[zh]
                child.parent_move_dict[self] = move
            else:
                child = Node(move, self)
                Node.hash_table[zh] = child

            self.children.append(child)
            state.pop()

    def update(self, result):
        self.visits += 1
        self.score += result

    def is_leaf(self):
        return self.children is None

    def has_parent(self):
        return len(self.parent_move_dict) != 0

    def get_move(self, parent):
        return self.parent_move_dict[parent]

    def get_parents(self):
        return self.parent_move_dict.keys()
