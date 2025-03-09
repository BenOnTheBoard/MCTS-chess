class EvaluatorInterface:
    def evaluate(self, node, is_max_player):
        """
        Return a value given a child node.
        """
        raise NotImplementedError()
