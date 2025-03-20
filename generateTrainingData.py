import chess
import chess.engine
from random import choice
from tqdm import tqdm

RAND_START_DEPTH = 6
SF_SEARCH_DEPTH = 24
MAX_PLIES = 90
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


def sf_self_play(board, data_file):
    while not board.is_game_over() and board.ply() < MAX_PLIES:
        eval_dict = stockfish.analyse(board, chess.engine.Limit(depth=SF_SEARCH_DEPTH))
        sf_eval = eval_dict["score"].white().score()
        sf_move = eval_dict["pv"][0]
        board.push(sf_move)

        if sf_eval is not None:
            data_file.write(f"{board.fen()}, {sf_eval}\n")


def generate(games):
    with open(DATA_FILENAME, "a") as data_file:
        for _ in tqdm(range(games)):
            board = setup_board()
            sf_self_play(board, data_file)


if __name__ == "__main__":
    generate(1000)
