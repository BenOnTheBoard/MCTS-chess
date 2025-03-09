import chess

from engine.mcts import MCTS

# Currently versus self
# since we don't have adjustable
# parameters or different models.

board = chess.Board()
mcts = MCTS(board, 0.5)

while not mcts.position.is_game_over():
    print()
    print(mcts.position)
    choice = mcts.get_move()
    mcts.add_move(choice)
