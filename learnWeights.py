import chess
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine.heuristics.networks.deeperConvNetwork import DeeperConvNetwork


class ChessDataset(Dataset):
    def __init__(self, filename, conversion):
        self.conversion = conversion
        self.data = []
        with open(filename, "r") as file:
            self.data = file.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        fen, y = line.strip().split(",")
        line_state = chess.Board(fen=fen)
        line_tsr = self.conversion(line_state)

        y = torch.as_tensor(
            [int(y)],
            dtype=torch.float,
        )
        y = torch.sigmoid(y)

        return line_tsr, y


def process_batch(network, batch, loss_fn):
    board_tensor, targets = batch
    board_tensor = board_tensor.view(len(board_tensor), 12, 8, 8)

    predictions = network.model(board_tensor)
    loss = loss_fn(predictions, targets)

    return loss


def main():
    # model = torch.load("saved_model.pt", weights_only=False)
    network_type = DeeperConvNetwork
    network = network_type()

    data_filename = "LesserTDRand.txt"  # "training_data.txt"
    tests_filename = "LesserTestData.txt"  # "testing_data.txt"
    output_filename = "new_saved_model.pt"
    loss_fn = torch.nn.MSELoss()
    rounds = 20
    learning_rate = 0.001
    batch_size = 4000

    dataset = ChessDataset(data_filename, network_type.board_to_tensor)
    testset = ChessDataset(tests_filename, network_type.board_to_tensor)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    best_loss = float("inf")

    for round in range(rounds):
        total_loss = 0
        for batch in tqdm(train_loader):
            loss = process_batch(network, batch, loss_fn)
            total_loss += loss.item()

            network.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in network.model.parameters():
                    param -= learning_rate * param.grad

        test_loss = 0
        for batch in test_loader:
            loss = process_batch(network, batch, loss_fn)
            test_loss += loss.item()

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(network.model, output_filename)
        else:
            break

        print(
            f"Round: {round}\tAvg. Training Loss: {total_loss / len(train_loader):.4f}\t\tAvg. Test Loss: {test_loss / len(test_loader):.4f}"
        )


if __name__ == "__main__":
    main()
