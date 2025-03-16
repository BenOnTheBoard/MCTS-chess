import chess
import requests

from engine.heuristics.tableBased.pieceTable import PieceTable
from engine.mcts import MCTS
from engine.treeEvaluators.UCT import UCT


# Versus Stockfish
def extract_stockfish_move(response):
    move_text = response.json()["bestmove"].split()[1]
    move = chess.Move.from_uci(move_text)
    return move


url = "https://stockfish.online/api/s/v2.php"
parameters = {"fen": None, "depth": 12}


board = chess.Board()

mcts = MCTS(board, 20, UCT(0.4), PieceTable())
while not mcts.position.is_game_over():
    white_choice = mcts.get_move()
    print(mcts.root_node.visits)
    print("Move analysis:")
    for child in mcts.root_node.children:
        print(
            f"{child.move.uci()}\t{child.score:.2f}\t{child.visits}\t{(child.score / child.visits):.2f}"
        )
    mcts.add_move(white_choice)

    print()
    print(mcts.position)
    break

    parameters["fen"] = mcts.position.fen()
    response = requests.get(url, params=parameters)
    black_choice = extract_stockfish_move(response)
    mcts.add_move(black_choice)

    print()
    print(black_choice)
    print(mcts.position)
