class Node:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.children = None
        self.score = 0
        self.visits = 0

    def expand_node(self, state):
        if self.children is None and not state.is_game_over():
            self.children = [Node(move, self) for move in state.legal_moves]

    def update(self, result):
        self.visits += 1
        self.score += result

    def is_leaf(self):
        return self.children is None

    def has_parent(self):
        return self.parent is not None
