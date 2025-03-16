from copy import deepcopy

import chess
import chess.engine
import numpy as np
from random import choice
from tqdm import tqdm

from engine.heuristics.tableBased.pieceTable import PieceTable
from engine.heuristics.tableBased.tables import TABLES

L_RATE = 0.001
ROUNDS = 5000
RAND_START_DEPTH = 6
SF_SEARCH_DEPTH = 8
stockfish = chess.engine.SimpleEngine.popen_uci(
    r"stockfish\stockfish-windows-x86-64-avx2.exe"
)
tables = deepcopy(TABLES)


def setup_board():
    start = chess.Board()
    for _ in range(RAND_START_DEPTH):
        if not start.is_game_over():
            start.push(choice(list(start.generate_legal_moves())))
    return start


def sf_self_play(board):
    game_history = [(chess.Board().fen(), 0, 0)]
    while not board.is_game_over() and len(board.piece_map()) > 8:
        eval_dict = stockfish.analyse(board, chess.engine.Limit(depth=SF_SEARCH_DEPTH))
        sf_eval = eval_dict["score"].white().score()
        sf_move = eval_dict["pv"][0]
        board.push(sf_move)

        if sf_eval is not None:
            game_history.append(
                (board.fen(), PieceTable(tables).evaluate(board), sf_eval)
            )
    return game_history


for round in tqdm(range(ROUNDS)):
    board = setup_board()
    game_history = sf_self_play(board)

    for fen, v, r in game_history:
        l_state = chess.Board(fen=fen)
        for square in chess.SQUARES:
            row, col = divmod(square, 8)
            piece = l_state.piece_at(square)
            if piece is not None:
                piece_table = tables[piece.piece_type]
                l_step = L_RATE * (r - v)
                if piece.color == chess.WHITE:
                    piece_table[7 - row][col] += l_step
                elif piece.color == chess.BLACK:
                    piece_table[row][col] -= l_step

np.set_printoptions(precision=0, suppress=True)
for table in tables.values():
    print(table)
    print()
