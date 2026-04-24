import bulletchess
import torch

from engine.heuristics.MatDHNetwork import MaterialDualHeadNetwork
from engine.mcts import MCTS
from engine.treeEvaluators.AlphaPUCT import AlphaPUCT


def pprint_children_recursive(node, limit=1_000, depth=0):
    spacer = "  " * depth
    for child in node.children:
        if child.visits >= limit:
            print(f"{spacer}{child.move}")
            pprint_children_recursive(child, limit, depth + 1)


def pprint_children(node):
    for child in node.children:
        print(f"\t{child.move}\t{child.prior:.2f}\t{child.quality:.2f}\t{child.visits}")


def pprint_principal_variation(node, limit=1_000):
    cur_node = node
    moves_uci = []
    while cur_node.children and cur_node.visits > limit:
        cur_node.children.sort(key=lambda n: n.visits)
        cur_node = cur_node.children[-1]
        moves_uci.append(cur_node.move.uci())
    print(" ".join(moves_uci))


def main():
    model = torch.load("models/mat_dhn.pt", weights_only=False, map_location="cpu")
    network = MaterialDualHeadNetwork(model)
    NODE_LIMIT = 10_000

    board = bulletchess.Board()
    mcts = MCTS(board, AlphaPUCT(2), network)

    while True:
        move = mcts.get_move(node_count=NODE_LIMIT, tqdm_on=True)
        pprint_children(mcts.root_node)

        mcts.add_move(move)
        print(f"\n{mcts.position}")

        if mcts.position in bulletchess.CHECKMATE:
            return
        if mcts.position in bulletchess.DRAW:
            return
        if abs(mcts.root_node.quality) > 0.95:
            return


if __name__ == "__main__":
    main()
