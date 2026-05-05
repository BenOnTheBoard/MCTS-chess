class Node:
    def __init__(self, move, prior):
        self.move = move
        self.children = None
        self.prior = prior
        self.quality = 0
        self.visits = 0

    def update_quality(self, value):
        self.visits += 1
        self.quality += (value - self.quality) / self.visits
