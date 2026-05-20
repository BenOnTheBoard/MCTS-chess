from bulletchess import Board
from torch import load as torch_load
from math import tanh
from multiprocessing import Pool
from random import choices, gauss
from statistics import mean
from time import perf_counter
from tqdm import tqdm

from engine.heuristics.MatDHNetwork import MaterialDualHeadNetwork
from engine.mcts import MCTS

# General
NETWORK = MaterialDualHeadNetwork(
    torch_load("models/mat_dhn.pt", weights_only=False, map_location="cpu")
)
NETWORK.model.eval()
POP = 250
NODE_LIMIT = 10_000
BATCH = 20
PROCESSES = 10
EPOCHS = 100
STEP = 0.01

# Worker
WORKER_MCTS = None


def read_file(filename):
    data = []
    with open(filename, "r") as file:
        for line in tqdm(file.readlines(), desc="File Reading"):
            fen, _, value_str = line.strip().split(",")
            value = tanh(int(value_str))
            data.append((fen, value))
    return data


def mutate(constants, step):
    return tuple(c + gauss(0, 1) * step for c in constants)


def worker_init():
    global WORKER_MCTS
    WORKER_MCTS = MCTS(Board(), NETWORK, constants=(0, 0))


def batch_worker(args):
    arms, fen, value = args
    pos = Board.from_fen(fen)
    losses = []
    for a in arms:
        WORKER_MCTS.set_constants(a)
        WORKER_MCTS.set_position(pos)
        WORKER_MCTS.get_move(NODE_LIMIT)
        losses.append(abs(value - WORKER_MCTS.root_node.quality))
    return losses


def main():
    data_filename = "data/DHTest.txt"
    data = read_file(data_filename)

    arms = [(mutate((1.4, 3), 0.5)) for _ in range(POP)]

    with Pool(PROCESSES, initializer=worker_init) as pool:
        for epoch in range(EPOCHS):
            start = perf_counter()
            jobs = [(arms, fen, value) for fen, value in choices(data, k=BATCH)]

            results = pool.map(batch_worker, jobs)
            scores = [sum(losses) for losses in zip(*results)]

            ranking = sorted(zip(arms, scores), key=lambda x: x[1])
            arms = [a for a, _ in ranking[: len(ranking) // 2]]

            new_arms = [mutate(a, STEP) for a in arms]
            arms.extend(new_arms)
            end = perf_counter()

            print(f"\nEpoch {epoch + 1}:\n----------")
            for i, a in enumerate(arms[:10]):
                print(f"{i + 1}: ({a[0]:.3f}, {a[1]:.3f})")
            print(f"Mean Loss: {mean(scores) / BATCH}")
            mins = (end - start) / 60
            print(f"Time: {int(mins)} m {60 * (mins % 1):.2f} s")


if __name__ == "__main__":
    main()
