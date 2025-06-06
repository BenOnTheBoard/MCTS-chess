import chess
import chess.engine
from random import choice
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

RAND_START_DEPTH = 6
SF_SEARCH_DEPTH = 10
MAX_PLIES = 150
STOCKFISH_PATH = r"stockfish\stockfish-windows-x86-64-avx2.exe"
DATA_FILENAME = "data/LesserTDTesting.txt"


def sf_analysis(engine, board):
    if not board.is_game_over():
        eval_dict = engine.analyse(board, chess.engine.Limit(depth=SF_SEARCH_DEPTH))
        sf_eval = eval_dict["score"].white().score()
        if sf_eval is not None:
            return f"{board.fen()}, {sf_eval}\n"
    return None


def generate_single_game(engine):
    board = chess.Board()
    lines = []
    while not board.is_game_over() and board.ply() < MAX_PLIES:
        rand_move = choice(list(board.legal_moves))
        board.push(rand_move)
        line = sf_analysis(engine, board)
        if line:
            lines.append(line)
    return lines


def worker(args):
    games, stockfish_path, temp_filename = args
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        with open(temp_filename, "w") as f:
            for _ in range(games):
                for line in generate_single_game(engine):
                    f.write(line)


def parallel_generate(total_games, n_workers=None):
    if n_workers is None:
        n_workers = cpu_count()
    games_per_worker = total_games // n_workers
    temp_files = [f"data/temp_worker_{i}.txt" for i in range(n_workers)]
    args = [(games_per_worker, STOCKFISH_PATH, temp_files[i]) for i in range(n_workers)]

    with Pool(n_workers) as pool:
        list(tqdm(pool.imap_unordered(worker, args), total=n_workers))

    # Merge temp files
    with open(DATA_FILENAME, "a") as outfile:
        for fname in temp_files:
            with open(fname) as infile:
                outfile.write(infile.read())
            os.remove(fname)


if __name__ == "__main__":
    parallel_generate(2_000)
