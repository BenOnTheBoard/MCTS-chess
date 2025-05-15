import chess
import chess.engine
import torch

from engine.heuristics.networks.dualPathNetwork import DualPathNetwork
from engine.mcts import MCTS
from engine.treeEvaluators.UCT import UCT
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
        print(f"{child.move.uci()}\t{child.visits}\t{(child.score / child.visits):.2f}")


def print_principal_variation(mcts, is_white):
    node = mcts.root_node
    moves = []
    turn = is_white
    while not node.is_leaf():
        node.children.sort(key=node_comparator)
        if turn:
            node = node.children[-1]
        else:
            node = node.children[0]
        moves.append(node.move)
        turn = not turn

    print(mcts.position.variation_san(moves))


if __name__ == "__main__":
    stockfish = chess.engine.SimpleEngine.popen_uci(
        r"stockfish\stockfish-windows-x86-64-avx2.exe"
    )

    board = chess.Board()

    model = torch.load("models/dpn.pt", weights_only=False)
    model.eval()

    mcts = MCTS(board, 40, UCT(3), DualPathNetwork(model))
    while not mcts.position.is_game_over():
        white_choice = mcts.get_move()
        print_principal_variation(mcts, True)
        print_analysis(mcts)
        mcts.add_move(white_choice)

        print()
        print(mcts.position)

        black_choice = stockfish.play(mcts.position, chess.engine.Limit(time=0.1)).move
        mcts.add_move(black_choice)

        print()
        print(black_choice)
        print(mcts.position)
