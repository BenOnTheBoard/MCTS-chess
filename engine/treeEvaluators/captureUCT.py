from math import log, sqrt

from engine.treeEvaluators.evaluatorInterface import EvaluatorInterface


class CaptureUCT(EvaluatorInterface):
    def __init__(self, C, capture_bonus):
        self.C = C
        self.capture_bonus = capture_bonus

    def prior(self, child, state):
        if state.is_capture(child.move):
            return 1 + self.capture_bonus
        return 1

    def evaluate(self, child, node, state):
        p = self.prior(child, state)
        exploring_term = self.C * p * sqrt(log(node.visits) / child.visits)

        if node.turn:
            return child.quality + exploring_term
        else:
            return child.quality - exploring_term
