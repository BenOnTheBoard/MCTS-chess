import chess
import chess.engine
import torch

from engine.heuristics.networks.convNetwork import ConvNetwork
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

model = torch.load("models/cnn.pt", weights_only=False)
model.eval()

mcts = MCTS(board, 60, UCT(1.5), ConvNetwork(model))
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
