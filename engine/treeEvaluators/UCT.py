from math import log, sqrt

from engine.treeEvaluators.evaluatorInterface import EvaluatorInterface


class UCT(EvaluatorInterface):
    def __init__(self, C):
        self.C = C

    def evaluate(self, child, node, is_max_player):
        child_quality = child.score / child.visits
        exploring_term = self.C * sqrt(log(node.visits) / child.visits)

        if is_max_player:
            return child_quality + exploring_term
        else:
            return child_quality - exploring_term
