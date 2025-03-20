import chess
import torch

from engine.heuristics.heuristicInterface import HeuristicInterface


class BasicNetwork(HeuristicInterface):
    def __init__(self):
        self.init_learning_rate = 0.1

        # layers_lens: 2 * 6 * 64, 3 * 64, 3 * 64, 1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(768, 192),
            torch.nn.SiLU(),
            torch.nn.Linear(192, 192),
            torch.nn.SiLU(),
            torch.nn.Linear(192, 1),
        )

    def int_to_bit_vector(self, num):
        bit_list = [int(bit) for bit in bin(num)[2:].zfill(64)]
        bit_vector = torch.tensor(bit_list, dtype=torch.float)
        return bit_vector

    def evaluate(self, state):
        input_sections = []
        for colour in chess.COLORS:
            for piece in chess.PIECE_TYPES:
                pc_int = state.pieces_mask(piece, colour)
                section = self.int_to_bit_vector(pc_int)
                input_sections.append(section)

        input_vector = torch.concatenate(input_sections)
        output_vector = self.model(input_vector)
        return output_vector.item()


print(BasicNetwork().evaluate(chess.Board()))
