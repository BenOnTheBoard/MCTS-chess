class Node:
    def __init__(self, move, prior):
        self.move = move
        self.children = None
        self.prior = prior
        self.quality = 0
        self.visits = 0
        self.variance = 2.5 * 10 ** (-4)

    def update(self, value):
        self.visits += 1
        d1 = value - self.quality
        self.quality += d1 / self.visits
        d2 = value - self.quality
        self.variance += (d1 * d2 - self.variance) / self.visits
