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
        board_tensor = torch.zeros((11, 64), dtype=torch.int8)
        p_map = state.piece_map()
        for square, piece in p_map.items():
            layer = piece.piece_type - 1
            if piece.color:
                board_tensor[layer, square] = 1
            else:
                board_tensor[layer, square] = -1

        if state.has_kingside_castling_rights(chess.WHITE):
            board_tensor[6, :] = 1
        if state.has_queenside_castling_rights(chess.WHITE):
            board_tensor[7, :] = 1
        if state.has_kingside_castling_rights(chess.BLACK):
            board_tensor[8, :] = 1
        if state.has_queenside_castling_rights(chess.BLACK):
            board_tensor[9, :] = 1
        if state.turn:
            board_tensor[10, :] = 1

        return board_tensor

    def tensor_eval(self, state):
        input_vector = self.board_to_tensor(state).to(torch.float32)
        output_vector = self.model(input_vector)
        return output_vector

    def evaluate(self, state):
        outcome = state.outcome()
        if outcome is not None:
            winner = outcome.winner
            return OUTCOMES[winner]

        output_vector = self.tensor_eval(state)
        return output_vector.item()


if __name__ == "__main__":
    from time import perf_counter_ns as ns
    from tqdm import tqdm
    from statistics import mean

    times = []
    b = chess.Board()
    for _ in tqdm(range(10_000)):
        start = ns()
        AbstractNetwork.board_to_tensor(b)
        end = ns()

        times.append(end - start)

    print(f"{mean(times) / 1000}μs")
