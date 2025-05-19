from engine.backpropagationRules.ruleInterface import RuleInterface


class MaxChild(RuleInterface):
    def calculate(self, node, value):
        new_quality = max(node.quality, value)
        return new_quality, new_quality
