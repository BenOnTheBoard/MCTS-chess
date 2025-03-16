import chess
from copy import deepcopy
import numpy as np

from engine.heuristics.tableBased.pieceTable import PieceTable
from engine.mcts import MCTS
from engine.heuristics.tableBased.tables import TABLES
from engine.treeEvaluators.UCT import UCT
from engine.values import OUTCOMES

L_RATE = 2

tables = deepcopy(TABLES)

for round in range(10):
    print(round)
    dataset = dict()

    game_history = []
    mcts = MCTS(chess.Board(), 1, UCT(1.5), PieceTable(tables))

    while not mcts.position.is_game_over():
        white_choice = mcts.get_move()
        mcts.add_move(white_choice)

        game_history.append(
            (
                mcts.position.fen(),
                PieceTable().evaluate(mcts.position),
            )
        )

    result = OUTCOMES[mcts.position.outcome().winner]

    for fen, prediction in game_history:
        dataset[fen] = (prediction, result)

    for fen, data in dataset.items():
        l_state = chess.Board(fen=fen)
        v = data[0]
        r = data[1]
        for square in chess.SQUARES:
            row, col = divmod(square, 8)
            piece = l_state.piece_at(square)
            if piece is not None:
                piece_table = tables[piece.piece_type]
                l_step = L_RATE * (r - v) * v * (1 - v)
                if l_state.turn == chess.WHITE:
                    piece_table[7 - row][col] += l_step
                elif l_state.turn == chess.BLACK:
                    piece_table[row][col] += l_step

np.set_printoptions(precision=3)
for table in tables.values():
    print(table)
    print()
