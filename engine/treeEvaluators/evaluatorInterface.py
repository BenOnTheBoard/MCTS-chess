class EvaluatorInterface:
    def evaluate(self, child, node, state):
        """
        Return a value given a child node.
        """
        raise NotImplementedError()
