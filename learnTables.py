from copy import deepcopy

import chess
import chess.engine
import numpy as np

from engine.heuristics.tableBased.pieceTable import PieceTable
from engine.heuristics.tableBased.tables import TABLES

L_RATE = 200
stockfish = chess.engine.SimpleEngine.popen_uci(
    r"stockfish\stockfish-windows-x86-64-avx2.exe"
)

tables = deepcopy(TABLES)

for round in range(1000):
    print(round)
    board = chess.Board()

    game_history = []
    while not board.is_game_over():
        result = stockfish.play(board, chess.engine.Limit(depth=20))
        board.push(result.move)
        game_history.append(
            (
                board.fen(),
                PieceTable(tables).evaluate(board),
                stockfish.analyse(board, chess.engine.Limit(depth=20)),
            )
        )

    for fen, v, r in game_history:
        print(f"{fen}\t\t{v}\t{r}")
        l_state = chess.Board(fen=fen)
        for square in chess.SQUARES:
            row, col = divmod(square, 8)
            piece = l_state.piece_at(square)
            if piece is not None:
                piece_table = tables[piece.piece_type]
                l_step = L_RATE * (r - v) * v * (1 - v)
                if l_state.turn == chess.WHITE:
                    piece_table[7 - row][col] += l_step
                elif l_state.turn == chess.BLACK:
                    piece_table[row][col] -= l_step

    if round % 10 == 0:
        np.set_printoptions(precision=3)
        for table in tables.values():
            print(table)
            print()

    quit()
