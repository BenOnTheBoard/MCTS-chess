from bulletchess import WHITE
from math import sqrt

from engine.treeEvaluators.evaluatorInterface import EvaluatorInterface


class AlphaPUCT(EvaluatorInterface):
    def __init__(self, C):
        self.C = C

    def evaluate(self, child, node):
        exploring_term = sqrt(node.visits) / (1 + child.visits)
        delta = self.C * child.prior * exploring_term

        if node.turn is WHITE:
            return child.quality + delta
        else:
            return child.quality - delta
