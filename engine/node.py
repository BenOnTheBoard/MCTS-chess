class Node:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.children = None
        self.quality = 0
        self.visits = 0

    def expand_node(self, state):
        if self.children is None and state.outcome() is None:
            self.children = [Node(move, self) for move in state.legal_moves]

    def update_quality(self, quality):
        if quality > 1 or quality < 0:
            raise ValueError("Qualities must be between 0 and 1.")
        self.quality = quality
        self.visits += 1

    def is_leaf(self):
        return self.children is None

    def has_parent(self):
        return self.parent is not None
