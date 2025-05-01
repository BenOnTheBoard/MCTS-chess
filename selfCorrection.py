import chess
import chess.engine
import torch

from engine.heuristics.networks.convNetwork import ConvNetwork
from engine.mcts import MCTS
from engine.treeEvaluators.UCT import UCT


def process_batch(model, batch, loss_fn):
    board_tensor, targets = batch
    board_tensor = board_tensor.to(torch.float32)
    board_tensor = board_tensor.view(len(board_tensor), 6, 8, 8)

    predictions = model(board_tensor)
    loss = loss_fn(predictions, targets)

    return loss


def train_epoch(model, batch, loss_fn, l_rate):
    model.train()
    loss = process_batch(model, batch, loss_fn)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= l_rate * param.grad
    model.eval()


def make_batch(mcts, network):
    variation_tensor_list = []
    target_tensor_list = []

    for child in mcts.root_node.children:
        mcts.position.push(child.move)

        variation = network.board_to_tensor(mcts.position)
        target = torch.as_tensor([child.score / child.visits], dtype=torch.float32)

        variation_tensor_list.append(variation)
        target_tensor_list.append(target)

        mcts.position.pop()

    board_tensor = torch.concatenate(variation_tensor_list).to(torch.float32)
    target_tensor = torch.concatenate(target_tensor_list).to(torch.float32).view(-1, 1)

    return (board_tensor, target_tensor)


def print_analysis(node, intuition):
    print("Move analysis:")
    for i, child in enumerate(node.children):
        ratio = child.score / child.visits
        move = child.move.uci()
        intuited = intuition[i].item()
        print(f"{move}\t{child.visits}\t{ratio:.3f}\t{intuited:.3f}")


if __name__ == "__main__":
    loss_fn = torch.nn.MSELoss()
    output_filename = "models/sc_only_cnn.pt"
    l_rate = 1e-1
    games = 12

    board = chess.Board()

    model = None  # torch.load("models/cnn.pt", weights_only=False)
    network = ConvNetwork(model)
    mcts = MCTS(board, 5, UCT(10000), network)

    for _ in range(games):
        while mcts.position.outcome(claim_draw=True) is None:
            choice = mcts.get_move()

            batch = make_batch(mcts, network)

            print_analysis(mcts.root_node, network.model(batch[0]))
            train_epoch(network.model, batch, loss_fn, l_rate)

            mcts.add_move(choice)
            # now wipe memory of analysis
            mcts.set_position(mcts.position)

            print()
            print(mcts.position)

            torch.save(network.model, output_filename)
