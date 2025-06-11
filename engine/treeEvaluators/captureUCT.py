from math import log, sqrt

from engine.treeEvaluators.evaluatorInterface import EvaluatorInterface


class CaptureUCT(EvaluatorInterface):
    def __init__(self, C, capture_bonus):
        self.C = C
        self.capture_bonus = capture_bonus

    def prior(self, node, state):
        if state.is_capture(node.move):
            return 1 + self.capture_bonus
        return 1

    def evaluate(self, child, node, state):
        p = self.prior(node, state)
        exploring_term = self.C * p * sqrt(node.visits / (1 + child.visits))

        if node.turn:
            return child.quality + exploring_term
        else:
            return child.quality - exploring_term
