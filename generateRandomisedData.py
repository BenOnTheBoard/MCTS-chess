import chess
import chess.engine
from random import choice
from tqdm import tqdm

RAND_START_DEPTH = 6
SF_SEARCH_DEPTH = 24
MAX_PLIES = 100
stockfish = chess.engine.SimpleEngine.popen_uci(
    r"stockfish\stockfish-windows-x86-64-avx2.exe"
)
DATA_FILENAME = "training_data_rand.txt"


def sf_analysis(board, data_file):
    if not board.is_game_over():
        eval_dict = stockfish.analyse(board, chess.engine.Limit(depth=SF_SEARCH_DEPTH))
        sf_eval = eval_dict["score"].white().score()
        if sf_eval is not None:
            data_file.write(f"{board.fen()}, {sf_eval}\n")


def generate(games):
    for _ in tqdm(range(games)):
        with open(DATA_FILENAME, "a") as data_file:
            board = chess.Board()
            while not board.is_game_over() and board.ply() < MAX_PLIES:
                rand_move = choice(list(board.legal_moves))
                board.push(rand_move)
                sf_analysis(board, data_file)


if __name__ == "__main__":
    generate(1000)
