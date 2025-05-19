"""
Standard MCTS Backpropagation
"""

from engine.backpropagationRules.ruleInterface import RuleInterface


class MeanChild(RuleInterface):
    def calculate(self, node, value):
        new_quality = node.quality + (value - node.quality) / (node.visits + 1)
        # in mean chlid you propagate the same value back up through all nodes
        return new_quality, value
