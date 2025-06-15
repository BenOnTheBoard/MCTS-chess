import torch

from engine.heuristics.networks.deltaOne import DeltaOne
from engine.treeEvaluators.UCT import UCT
from engine.backpropagationRules.meanChild import MeanChild

from matches import play_single_game


def main():
    model_delta = torch.load("models/new_delta.pt", weights_only=False)
    model_delta.eval()

    NODES_PER_MOVE = 25_000

    player_config = (
        UCT(1.3),
        DeltaOne(model_delta),
        MeanChild(),
    )

    play_single_game(
        player_config,
        player_config,
        NODES_PER_MOVE,
        start_depth=0,
        verbose=True,
    )


if __name__ == "__main__":
    main()
