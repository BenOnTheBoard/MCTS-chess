from copy import deepcopy

import chess
import chess.engine
from math import log10
import numpy as np
from random import choice
from tqdm import tqdm

from engine.heuristics.tableBased.pieceTable import PieceTable
from engine.heuristics.tableBased.tables import BLANK_TABLES, TABLES

RAND_START_DEPTH = 6
SF_SEARCH_DEPTH = 20
stockfish = chess.engine.SimpleEngine.popen_uci(
    r"stockfish\stockfish-windows-x86-64-avx2.exe"
)
DATA_FILENAME = "training_data.txt"


def setup_board():
    start = chess.Board()
    for _ in range(RAND_START_DEPTH):
        if not start.is_game_over():
            start.push(choice(list(start.generate_legal_moves())))
    return start


def sf_self_play(board):
    while not board.is_game_over() and board.ply() < 80:
        eval_dict = stockfish.analyse(board, chess.engine.Limit(depth=SF_SEARCH_DEPTH))
        sf_eval = eval_dict["score"].white().score()
        sf_move = eval_dict["pv"][0]
        board.push(sf_move)

        if sf_eval is not None:
            with open(DATA_FILENAME, "a") as data_file:
                data_file.write(f"{board.fen()}, {sf_eval}\n")


def generate(games):
    for _ in tqdm(range(games)):
        board = setup_board()
        sf_self_play(board)


def train(tables, l_rate):
    model = PieceTable(tables)
    changes = deepcopy(BLANK_TABLES)
    error = 0
    with open(DATA_FILENAME, "r") as data_file:
        dataset = data_file.readlines()
        for line in dataset:
            fen, r = line.strip().split(",")
            r = int(r)
            l_state = chess.Board(fen=fen)
            v = model.evaluate(l_state)
            error += (r - v) ** 2

            for square in chess.SQUARES:
                row, col = divmod(square, 8)
                piece = l_state.piece_at(square)
                if piece is not None:
                    piece_table = changes[piece.piece_type]
                    l_step = l_rate * (r - v)
                    if piece.color == chess.WHITE:
                        piece_table[7 - row][col] += l_step
                    elif piece.color == chess.BLACK:
                        piece_table[row][col] -= l_step

    for piece, table in tables.items():
        changes[piece] /= len(dataset)
        table += changes[piece]

    error /= len(dataset)
    return error


def pprint_tables(tables):
    np.set_printoptions(precision=0, suppress=True)
    for table in tables.values():
        print(table)
        print()


def main():
    l_rate = 2
    rounds = 2_000
    tables = deepcopy(TABLES)
    for round in tqdm(range(rounds)):
        error = train(tables, l_rate)
        print(f"\nRound: {round}, Error:{error} = 10 ^ {log10(error)}")
    pprint_tables(tables)


if __name__ == "__main__":
    main()
