from math import exp

from engine.backpropagationRules.ruleInterface import RuleInterface


class ApproxSoftMax(RuleInterface):
    def __init__(self, w):
        self.w = w

    def calculate(self, node, value):
        diff = value - node.quality
        if node.turn:
            weight = exp(-self.w * diff)  # softmax
        else:
            weight = exp(self.w * diff)  # softmin

        new_quality = node.quality + diff / (1 + node.visits * weight)

        return new_quality, value
