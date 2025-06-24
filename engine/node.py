class Node:
    def __init__(self, move, turn, prior, parent):
        self.move = move
        self.turn = turn
        self.parent = parent
        self.prior = prior
        self.children = None
        self.quality = 0
        self.visits = 0

    def update_quality(self, quality):
        if quality > 1 or quality < -1:
            raise ValueError("Qualities must be between -1 and 1.")
        self.quality = quality
        self.visits += 1

    def is_leaf(self):
        return self.children is None

    def has_parent(self):
        return self.parent is not None
