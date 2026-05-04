class Node:
    def __init__(self, move, turn, prior, parent):
        self.move = move
        self.turn = turn
        self.parent = parent
        self.prior = prior
        self.children = None
        self.quality = 0
        self.visits = 0

    def update_quality(self, value):
        self.visits += 1
        self.quality += (value - self.quality) / self.visits

    def is_leaf(self):
        return self.children is None

    def has_parent(self):
        return self.parent is not None
