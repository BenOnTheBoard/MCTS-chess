from math import exp, log, sqrt

from engine.treeEvaluators.evaluatorInterface import EvaluatorInterface


class PUCT(EvaluatorInterface):
    def __init__(self, C):
        self.C = C

    def evaluate(self, child, node):
        exploring_term = self.C * sqrt(log(node.visits) / child.visits)

        # Factor is 2/M where M is the probability derived from the logits
        # See Chris D. Rosin PUCB paper
        prior_factor = 2 * (1 + exp(-child.prior))
        prior_term = prior_factor * sqrt(log(node.visits) / node.visits)

        if node.turn:
            return child.quality + exploring_term - prior_term
        else:
            return child.quality - exploring_term + prior_term
