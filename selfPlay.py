import bulletchess
from bulletchess import QUEEN, PAWN, SQUARES, CHECKMATE, DRAW, PIECE_TYPES
from collections import deque
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine.backpropagationRules.meanChild import MeanChild
from engine.heuristics.dualHeadNetwork import DualHeadNetwork
from engine.mcts import MCTS
from engine.treeEvaluators.PUCT import PUCT
from engine.values import DIRECTIONS, KNIGHTS_MOVES


class PolicyBoardDataset(Dataset):
    def __init__(self, board_conversion, move_conversion, max_size=2**15):
        self.board_conversion = board_conversion
        self.move_conversion = move_conversion
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def add_triple(self, board, result, move):
        self.buffer.append((board, result, move))

    def __getitem__(self, idx):
        board, result, move = self.buffer[idx]
        board_tsr = self.board_conversion(board)
        y_move = torch.tensor(self.move_conversion(move), dtype=torch.long)
        y_value = torch.tensor(result, dtype=torch.float32)
        return board_tsr, y_value, y_move


def process_batch(network, batch, value_loss_fn, policy_loss_fn):
    board_tensor, target_values, target_moves = batch
    batch_size = len(board_tensor)

    board_tensor = board_tensor.to(dtype=torch.float32)
    board_tensor = board_tensor.view(batch_size, 11, 8, 8)

    predicted_values, predicted_moves = network.model(board_tensor)
    predicted_moves = predicted_moves.view(batch_size, 4672)
    predicted_values = predicted_values.view(-1)

    value_loss = value_loss_fn(predicted_values, target_values)
    policy_loss = policy_loss_fn(predicted_moves, target_moves)

    return value_loss, policy_loss


def tensor_to_move(board, distribution):
    flat_index = torch.multinomial(distribution.flatten(), num_samples=1).item()
    plane, location = divmod(flat_index, 64)
    row, col = divmod(location, 8)
    from_square = SQUARES[row * 8 + col]

    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        dr, dc = DIRECTIONS[dir_idx]
        to_row = row + dr * dist
        to_col = col + dc * dist

        if to_row in (0, 7) and board[from_square] == PAWN:
            promotion = QUEEN
        else:
            promotion = None

    elif plane < 64:
        knight_idx = plane - 56
        dr, dc = KNIGHTS_MOVES[knight_idx]
        to_row = row + dr
        to_col = col + dc
        promotion = None

    else:
        prom_plane = plane - 64
        prom_idx, dir_type = divmod(prom_plane, 3)
        if row == 1:
            to_row = 0
        elif row == 6:
            to_row = 7
        else:
            raise ValueError("Promotion has bad rank")
        to_col = col + dir_type - 1
        promotion = PIECE_TYPES[prom_idx + 1]

    to_square = SQUARES[to_row * 8 + to_col]

    return bulletchess.Move(from_square, to_square, promote_to=promotion)


def play_game(network):
    board = bulletchess.Board()
    history = []
    while True:
        value, distribution = network.get_masked_move_distribution(board)

        choice = tensor_to_move(board, distribution)

        history.append(board.copy())
        board.apply(choice)

        if abs(value) > 0.99:
            return history

        if board in CHECKMATE or board in DRAW:
            return history


def main():
    model = torch.load("models/dhn.pt", weights_only=False, map_location="cpu")
    network_type = DualHeadNetwork
    network = network_type(model=model)

    output_filename = "models/new_dhn.pt"
    value_loss_fn = torch.nn.MSELoss(reduction="sum")
    policy_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    games = 100_000
    init_learning_rate = 1e-3
    batch_size = 64

    visits_per_evaluation = 1000
    tree_evaluator = PUCT(1.225)
    backprop_rule = MeanChild()

    dataset = PolicyBoardDataset(
        network_type.board_to_tensor, network_type.move_to_flat_index
    )

    optimizer = torch.optim.Adam(network.model.parameters(), lr=init_learning_rate)

    print("Setup complete.")

    for game_id in range(1, games + 1):
        network.model.eval()
        history = play_game(network)

        mcts = MCTS(
            bulletchess.Board(),
            tree_evaluator,
            network,
            backprop_rule,
        )

        for board in tqdm(history, desc=f"Game {game_id} Review"):
            mcts.set_position(board)
            best_move = mcts.get_move(node_count=visits_per_evaluation)
            dataset.add_triple(board, mcts.root_node.quality, best_move)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        network.model.train()
        total_value_loss = 0
        total_policy_loss = 0
        for batch in tqdm(train_loader, desc=f"Training after game {game_id}"):
            optimizer.zero_grad()
            value_loss, policy_loss = process_batch(
                network, batch, value_loss_fn, policy_loss_fn
            )
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            composite_loss = value_loss + policy_loss
            composite_loss.backward()
            optimizer.step()

        if game_id % 25 == 0:
            print(f"""
                Games played: {game_id}
                Value Head Loss: {total_value_loss / len(dataset)}
                Policy Head Loss: {total_policy_loss / len(dataset)}
            """)

        torch.save(network.model, output_filename)

    print("\n\nTraining is done!")


if __name__ == "__main__":
    main()
