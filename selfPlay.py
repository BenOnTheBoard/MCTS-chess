import chess
from collections import deque
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine.heuristics.simpleResNetPolicy import SimpleResNetPolicy
from engine.values import DIRECTIONS, KNIGHTS_MOVES


class PolicyBoardDataset(Dataset):
    def __init__(self, board_conversion, move_conversion, max_size=1000):
        self.board_conversion = board_conversion
        self.move_conversion = move_conversion
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def add_pair(self, board, move):
        self.buffer.append((board, move))

    def __getitem__(self, idx):
        board, move = self.buffer[idx]
        board_tsr = self.board_conversion(board)
        y = self.move_conversion(move)
        return board_tsr, y


def process_batch(network, batch, loss_fn):
    board_tensor, targets = batch
    board_tensor = board_tensor.to(torch.float32)
    board_tensor = board_tensor.view(len(board_tensor), 11, 8, 8)

    predictions = network.model(board_tensor)
    loss = loss_fn(predictions, targets)

    return loss


def tensor_to_move(board, distribution):
    flat_index = torch.multinomial(distribution.flatten(), num_samples=1).item()
    plane, location = divmod(flat_index, 64)
    row, col = divmod(location, 8)
    from_square = row * 8 + col

    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        dr, dc = DIRECTIONS[dir_idx]
        to_row = row + dr * dist
        to_col = col + dc * dist

        if to_row in (0, 7) and board.piece_type_at(from_square) == chess.PAWN:
            promotion = chess.QUEEN
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
        to_row = row + 1
        to_col = col + dir_type - 1
        promotion = prom_idx + 2

    to_square = to_row * 8 + to_col
    return chess.Move(from_square, to_square, promotion=promotion)


def play_game(network):
    while True:
        board = chess.Board()
        history = {chess.WHITE: [], chess.BLACK: []}
        outcome = None
        while outcome is None:
            distribution = network.get_masked_move_distribution(board)
            choice = tensor_to_move(board, distribution)

            history[board.turn].append((board.copy(), choice))
            board.push(choice)

            outcome = board.outcome(claim_draw=True)

        if outcome.winner is not None:
            return history[outcome.winner]


def main():
    model = None
    network_type = SimpleResNetPolicy
    network = network_type(model=model)

    output_filename = "models/new_srnp.pt"
    loss_fn = torch.nn.BCELoss(reduction="sum")
    games = 100_000
    init_learning_rate = 1e-4
    batch_size = 32

    dataset = PolicyBoardDataset(
        network_type.board_to_tensor, network_type.move_to_tensor
    )

    optimizer = torch.optim.Adam(network.model.parameters(), lr=init_learning_rate)

    for game_id in range(games):
        network.model.eval()
        history = play_game(network)
        for board, move in history:
            dataset.add_pair(board, move)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        network.model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            loss = process_batch(network, batch, loss_fn)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Game: {game_id}\tTraining Loss: {total_loss / len(dataset)}")

        if game_id % 25 == 0:
            torch.save(network.model, output_filename)

    print("\n\nTraining is done!")


if __name__ == "__main__":
    main()
