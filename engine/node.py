class Node:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.children = []
        self.score = 0
        self.visits = 0

    def expand_node(self, state):
        if not state.is_game_over():
            for move in state.legal_moves:
                new_child = Node(move, self)
                self.children.append(new_child)

    def update(self, result):
        self.visits += 1
        self.score += result

    def is_leaf(self):
        return len(self.children) == 0

    def has_parent(self):
        return self.parent is not None
