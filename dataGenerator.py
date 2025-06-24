import chess
import chess.engine
from random import choice
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

SF_SEARCH_DEPTH = 8
MAX_PLIES = 150
GAMES = 32_000
STOCKFISH_PATH = r"stockfish\stockfish-windows-x86-64-avx2.exe"
DATA_FILENAME = "data/DHRand.txt"


def sf_analysis(engine, board):
    if not board.is_game_over():
        result = engine.analyse(board, chess.engine.Limit(depth=SF_SEARCH_DEPTH))
        best_move = result["pv"][0]
        score = result["score"].white().score(mate_score=60_000)
        return f"{board.fen()},{best_move},{score}\n"
    return None


def generate_single_game(engine):
    board = chess.Board()
    lines = []
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        rand_move = choice(list(board.legal_moves))
        board.push(rand_move)
        line = sf_analysis(engine, board)
        if line is not None:
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
    parallel_generate(GAMES, 6)
