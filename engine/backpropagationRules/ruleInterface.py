class RuleInterface:
    def calculate(self, node, value):
        """
        Given a node and a new value, return a new value for the quality and the next value to pass up.
        """
        raise NotImplementedError()
