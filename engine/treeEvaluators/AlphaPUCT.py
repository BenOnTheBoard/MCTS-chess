from bulletchess import WHITE
from math import exp, sqrt

from engine.treeEvaluators.evaluatorInterface import EvaluatorInterface


class AlphaPUCT(EvaluatorInterface):
    def __init__(self, C):
        self.C = C

    def evaluate(self, child, node):
        exploring_term = sqrt(node.visits) / (1 + child.visits)
        prior_prob = 1 / (1 + exp(-child.prior))

        delta = self.C * prior_prob * exploring_term

        if node.turn is WHITE:
            return child.quality + delta
        else:
            return child.quality - delta
