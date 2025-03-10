import chess
import numpy as np
import random

from engine.nodeEvaluators.UCT import UCT
from engine.mcts import MCTS
from engine.values import OUTCOMES

upper_bound = 4
lower_bound = 0
step = 0.5
SEC_PER_TURN = 30
TOTAL_GAMES = 7 * 6 * 100


def play_single_game(white, black):
    while True:
        white_choice = white.get_move()
        white.add_move(white_choice)
        if white.position.is_game_over():
            return OUTCOMES[white.position.outcome().winner]
        else:
            black.add_move(white_choice)

        black_choice = black.get_move()
        black.add_move(black_choice)
        if black.position.is_game_over():
            return OUTCOMES[black.position.outcome().winner]
        else:
            white.add_move(black_choice)


origin = chess.Board()

competitor_nums = list(np.arange(lower_bound, upper_bound, step))
table = {}
for c_val in competitor_nums:
    mcts = MCTS(origin, SEC_PER_TURN)
    mcts.node_evaluator = UCT(c_val)
    table[c_val] = {
        "bot": mcts,
        "score": 0,
        "played": 0,
    }

for game in range(TOTAL_GAMES):
    white_num, black_num = random.sample(competitor_nums, 2)

    white_player = table[white_num]["bot"]
    black_player = table[black_num]["bot"]
    table[white_num]["played"] += 1
    table[black_num]["played"] += 1

    white_player.set_position(origin)
    black_player.set_position(origin)
    result = play_single_game(white_player, black_player)
    print(game, white_num, black_num, result)

    table[white_num]["score"] += result
    table[black_num]["score"] += 1 - result

for k, v in table.items():
    print(f"Bot {k}: {v['score']}/{v['played']}")
