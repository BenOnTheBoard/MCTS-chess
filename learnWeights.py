import chess
from math import cos, exp, pi
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine.heuristics.networks.dualPathNetwork import DualPathNetwork


class ChessDataset(Dataset):
    def __init__(self, filename, conversion):
        self.conversion = conversion
        self.data = []
        with open(filename, "r") as file:
            content = file.readlines()

        for line in tqdm(content):
            fen, y = line.strip().split(",")
            line_state = chess.Board(fen=fen)
            line_tsr = self.conversion(line_state)
            mirror_line_tsr = self.conversion(line_state.mirror())

            y = torch.as_tensor(
                [int(y)],
                dtype=torch.float,
            )
            y = torch.sigmoid(y)

            self.data.append((line_tsr, y))
            self.data.append((mirror_line_tsr, 1 - y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def process_batch(network, batch, loss_fn):
    board_tensor, targets = batch
    board_tensor = board_tensor.to(torch.float32)
    board_tensor = board_tensor.view(len(board_tensor), 6, 8, 8)

    predictions = network.model(board_tensor)
    loss = loss_fn(predictions, targets)

    return loss


def int_sigmoid(x):
    return 1 / (1 + exp(-x))


def learning_rate_function(start, ep, total_ep):
    return start * ((1 + cos(pi * ep / total_ep)) / 2)


def main():
    model = None
    network_type = DualPathNetwork
    network = network_type(model=model)

    data_filename = "data/LesserTDRand.txt"
    tests_filename = "data/LesserTestData.txt"
    output_filename = "models/new_cnn.pt"
    loss_fn = torch.nn.BCELoss()
    total_epochs = 100
    init_learning_rate = 1e-1
    batch_size = 4096

    dataset = ChessDataset(data_filename, network_type.board_to_tensor)
    testset = ChessDataset(tests_filename, network_type.board_to_tensor)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    for epoch in range(total_epochs):
        network.model.train()
        learning_rate = learning_rate_function(init_learning_rate, epoch, total_epochs)
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
                Epoch: {epoch}
                Learning Rate:{learning_rate:.6f}
                Avg. Training Loss: {total_loss / len(train_loader):.6f}
        """)

        network.model.eval()
        test_loss = 0
        for batch in test_loader:
            loss = process_batch(network, batch, loss_fn)
            test_loss += loss.item()

        print(f"\t\t Model test loss: {test_loss / len(test_loader):.4f}\n")
        torch.save(network.model, output_filename)

    print("\n\nTraining is done!")


if __name__ == "__main__":
    main()
