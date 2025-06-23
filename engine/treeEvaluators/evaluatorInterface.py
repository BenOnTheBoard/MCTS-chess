class EvaluatorInterface:
    def evaluate(self, child, node):
        """
        Return a value given a child node.
        """
        raise NotImplementedError()
