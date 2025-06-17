import chess
from multiprocessing import Pool
from random import choice
import torch
from tqdm import tqdm

from engine.mcts import MCTS
from engine.treeEvaluators.UCT import UCT
from engine.treeEvaluators.captureUCT import CaptureUCT
from engine.backpropagationRules.meanChild import MeanChild
from engine.utils import node_comparator
from engine.values import OUTCOMES


def print_analysis(mcts):
    print(mcts.root_node.visits)
    print("Move analysis:")
    for child in mcts.root_node.children:
        print(f"{child.move.uci()}\t{child.visits}\t{(child.quality):.2f}")


def print_principal_variation(mcts):
    node = mcts.root_node
    moves = []
    while not node.is_leaf():
        node.children.sort(key=node_comparator)
        if node.turn:
            node = node.children[-1]
        else:
            node = node.children[0]
        moves.append(node.move)

    print(mcts.position.variation_san(moves))


def get_new_starting_position(start_depth):
    board = chess.Board()
    for _ in range(start_depth):
        moves = list(board.legal_moves)
        if moves:
            rand_move = choice(moves)
            board.push(rand_move)
    return board


def play_single_game(
    white_args, black_args, nodes_per_move, start_depth, verbose=False
):
    board = get_new_starting_position(start_depth)
    white = MCTS(board, *white_args)
    black = MCTS(board, *black_args)

    while True:
        white_choice = white.get_move(nodes_per_move, verbose)
        if verbose:
            print_principal_variation(white)
            print_analysis(white)

        white.add_move(white_choice)
        black.add_move(white_choice)

        if verbose:
            print()
            print(white.position)

        outcome = white.position.outcome(claim_draw=True)
        if outcome is not None:
            return OUTCOMES[outcome.winner]

        black_choice = black.get_move(nodes_per_move, verbose)
        if verbose:
            print_principal_variation(black)
            print_analysis(black)

        white.add_move(black_choice)
        black.add_move(black_choice)

        if verbose:
            print()
            print(black.position)

        outcome = black.position.outcome(claim_draw=True)
        if outcome is not None:
            return OUTCOMES[outcome.winner]


def play_game_wrapper(args):
    NODES_PER_MOVE, RAND_START_DEPTH, player_one_config, player_two_config = args

    model_one = torch.load(player_one_config[0], weights_only=False)
    model_one.eval()
    model_two = torch.load(player_two_config[0], weights_only=False)
    model_two.eval()

    player_one_args = (
        player_one_config[2],
        player_one_config[1](model_one),
        player_one_config[3],
    )
    player_two_args = (
        player_two_config[2],
        player_two_config[1](model_one),
        player_two_config[3],
    )
    return play_single_game(
        player_one_args, player_two_args, NODES_PER_MOVE, RAND_START_DEPTH
    )


def main():
    model_delta = torch.load("models/new_delta.pt", weights_only=False)
    model_delta.eval()

    NODES_PER_MOVE = 2500
    RAND_START_DEPTH = 4  # ply
    GAMES = 100  # must be even

    player_one_config = (
        "models/new_delta.pt",
        None,
        CaptureUCT(1.5, 0.15),
        MeanChild(),
    )
    player_two_config = (
        "models/new_delta.pt",
        None,
        UCT(1.5),
        MeanChild(),
    )

    args_one_is_white = [
        (
            NODES_PER_MOVE,
            RAND_START_DEPTH,
            player_one_config,
            player_two_config,
        )
        for _ in range(GAMES // 2)
    ]

    args_one_is_black = [
        (
            NODES_PER_MOVE,
            RAND_START_DEPTH,
            player_two_config,
            player_one_config,
        )
        for _ in range(GAMES // 2)
    ]

    with Pool(6) as pool:
        white_results = list(
            tqdm(
                pool.imap_unordered(play_game_wrapper, args_one_is_white),
                total=GAMES // 2,
            ),
        )
        black_results = list(
            tqdm(
                pool.imap_unordered(play_game_wrapper, args_one_is_black),
                total=GAMES // 2,
            ),
        )

    results = white_results + [1 - r for r in black_results]

    print("Results:")
    print(f"\tPlayer One:\t{sum(results)}")
    print(f"\tPlayer Two:\t{GAMES - sum(results)}")
    print(f"\tScore:\t{(sum(results) / GAMES) * 100:.2f}%")


if __name__ == "__main__":
    main()
