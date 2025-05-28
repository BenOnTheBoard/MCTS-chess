import chess
import chess.engine
import torch

from engine.heuristics.networks.dualPathNetwork import DualPathNetwork
from engine.heuristics.networks.chessConvNetwork import ChessConvNetwork
from engine.mcts import MCTS
from engine.treeEvaluators.UCT import UCT
from engine.backpropagationRules.approxSoftMax import ApproxSoftMax
from engine.backpropagationRules.meanChild import MeanChild
from engine.utils import node_comparator


# Versus Stockfish
def extract_stockfish_move(response):
    move_text = response.json()["bestmove"].split()[1]
    move = chess.Move.from_uci(move_text)
    return move


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


if __name__ == "__main__":
    stockfish = chess.engine.SimpleEngine.popen_uci(
        r"stockfish\stockfish-windows-x86-64-avx2.exe"
    )

    board = chess.Board()

    model_dpn = torch.load("models/dpn.pt", weights_only=False)
    model_dpn.eval()
    model_ccnn = torch.load("models/ccnn.pt", weights_only=False)
    model_ccnn.eval()

    NODES_PER_MOVE = 100_000
    white = MCTS(board, UCT(2), ChessConvNetwork(model_ccnn), MeanChild())
    black = MCTS(board, UCT(2), DualPathNetwork(model_dpn), ApproxSoftMax(0.5))

    while not black.position.is_game_over():
        white_choice = white.get_move(NODES_PER_MOVE)
        print_principal_variation(white)
        print_analysis(white)

        white.add_move(white_choice)
        black.add_move(white_choice)

        print()
        print(white.position)

        if white.position.is_game_over():
            break

        black_choice = black.get_move(NODES_PER_MOVE)
        print_principal_variation(black)
        print_analysis(black)

        white.add_move(black_choice)
        black.add_move(black_choice)

        print()
        print(black.position)
