import chess
import chess.engine
import torch

from engine.heuristics.networks.convNetworkV2 import ConvNetworkV2
from engine.mcts import MCTS
from engine.treeEvaluators.UCT import UCT


# Versus Stockfish
def extract_stockfish_move(response):
    move_text = response.json()["bestmove"].split()[1]
    move = chess.Move.from_uci(move_text)
    return move


SF_SEARCH_DEPTH = 12
stockfish = chess.engine.SimpleEngine.popen_uci(
    r"stockfish\stockfish-windows-x86-64-avx2.exe"
)

board = chess.Board()

model = torch.load("models/V2_cnn.pt", weights_only=False)

mcts = MCTS(board, 20, UCT(2), ConvNetworkV2(model))
while not mcts.position.is_game_over():
    white_choice = mcts.get_move()
    print(mcts.root_node.visits)
    print("Move analysis:")
    for child in mcts.root_node.children:
        print(f"{child.move.uci()}\t{child.visits}\t{(child.score / child.visits):.2f}")
    mcts.add_move(white_choice)

    print()
    print(mcts.position)

    black_choice = stockfish.play(mcts.position, chess.engine.Limit(time=0.1)).move
    mcts.add_move(black_choice)

    print()
    print(black_choice)
    print(mcts.position)
