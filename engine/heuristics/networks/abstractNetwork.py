import chess
import torch

from engine.heuristics.heuristicInterface import HeuristicInterface
from engine.values import OUTCOMES


class AbstractNetwork(HeuristicInterface):
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.init_model()

    def init_model(self):
        raise NotImplementedError("Model forms must be supplied by subclass.")

    @staticmethod
    def board_to_tensor(state):
        board_tensor = torch.zeros((6, 64))
        for square in chess.SQUARES:
            piece = state.piece_at(square)
            if piece is None:
                continue

            layer = piece.piece_type - 1
            if piece.color is chess.WHITE:
                board_tensor[layer, square] = 1
            else:
                board_tensor[layer, square] = -1

        return board_tensor

    def tensor_eval(self, state):
        input_vector = self.board_to_tensor(state).to(torch.float32)
        output_vector = self.model(input_vector)
        return output_vector

    def evaluate(self, state):
        outcome = state.outcome(claim_draw=True)
        if outcome is not None:
            winner = outcome.winner
            return OUTCOMES[winner]

        output_vector = self.tensor_eval(state)
        return output_vector.item()


if __name__ == "__main__":
    from time import perf_counter_ns as ns
    from tqdm import tqdm
    from statistics import median

    times = []
    for _ in tqdm(range(100_000)):
        start = ns()
        AbstractNetwork.board_to_tensor(chess.Board())
        end = ns()

        times.append(end - start)

    print(f"{median(times) / 1000}μs")
