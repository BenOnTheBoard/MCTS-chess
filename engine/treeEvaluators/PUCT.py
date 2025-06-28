from math import log, sqrt

from engine.treeEvaluators.evaluatorInterface import EvaluatorInterface


class PUCT(EvaluatorInterface):
    def __init__(self, C):
        self.C = C

    def evaluate(self, child, node):
        exploring_term = self.C * sqrt(log(node.visits) / child.visits)
        prior_term = child.prior * sqrt(log(node.visits) / node.visits)

        if node.turn:
            return child.quality + exploring_term + prior_term
        else:
            return child.quality - exploring_term - prior_term
