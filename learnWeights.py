import chess
from math import exp
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine.heuristics.networks.convNetwork import ConvNetwork


class ChessDataset(Dataset):
    def __init__(self, filename, conversion):
        self.conversion = conversion
        self.data = []
        with open(filename, "r") as file:
            print("Reading in data...")
            for line in tqdm(file.readlines()):
                fen, y = line.strip().split(",")
                line_state = chess.Board(fen=fen)
                line_tsr = self.conversion(line_state)

                y = torch.as_tensor(
                    [int(y)],
                    dtype=torch.float,
                )
                y = torch.sigmoid(y)

                self.data.append((line_tsr, y))
            print("Data done.\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def process_batch(network, batch, loss_fn):
    board_tensor, targets = batch
    board_tensor = board_tensor.view(len(board_tensor), 12, 8, 8)

    predictions = network.model(board_tensor)
    loss = loss_fn(predictions, targets)

    return loss


def int_sigmoid(x):
    return 1 / (1 + exp(-x))


def learning_rate_function(start, n):
    a = 10
    return start / ((n / a) + 1)


def main():
    model = torch.load("models/conv_model.pt", weights_only=False)
    network_type = ConvNetwork
    network = network_type(model=model)

    data_filename = "data/LesserTDRand.txt"
    tests_filename = "data/LesserTestData.txt"
    output_filename = "models/new_conv_model.pt"
    loss_fn = torch.nn.MSELoss()
    rounds = 20
    init_learning_rate = 0.2
    batch_size = 8192

    dataset = ChessDataset(data_filename, network_type.board_to_tensor)
    testset = ChessDataset(tests_filename, network_type.board_to_tensor)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    for round in range(rounds):
        learning_rate = learning_rate_function(init_learning_rate, round)
        total_loss = 0
        for batch in tqdm(train_loader):
            loss = process_batch(network, batch, loss_fn)
            total_loss += loss.item()

            network.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in network.model.parameters():
                    param -= learning_rate * param.grad

        print(f"""
                Round: {round}
                Learning Rate:{learning_rate:.6f}
                Avg. Training Loss: {total_loss / len(train_loader):.6f}
        """)

    test_loss = 0
    for batch in test_loader:
        loss = process_batch(network, batch, loss_fn)
        test_loss += loss.item()

    print(f"Final model test loss: {test_loss / len(test_loader):.4f}")

    torch.save(model, output_filename)


if __name__ == "__main__":
    main()
