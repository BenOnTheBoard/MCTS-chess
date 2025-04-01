import chess
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine.heuristics.convNetwork import ConvNetwork


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


def process_batch(BNet, batch, loss_fn):
    board_tensor, targets = batch
    board_tensor = board_tensor.view(len(board_tensor), 12, 8, 8)

    predictions = BNet.model(board_tensor)
    loss = loss_fn(predictions, targets)

    return loss


def main():
    BNet = ConvNetwork()

    data_filename = "LesserTDRand.txt"  # "training_data.txt"
    tests_filename = "LesserTestData.txt"  # "testing_data.txt"
    output_filename = "saved_model.pt"
    loss_fn = torch.nn.MSELoss()
    rounds = 20
    learning_rate = 1
    batch_size = 4000

    dataset = ChessDataset(data_filename, ConvNetwork.board_to_tensor)
    testset = ChessDataset(tests_filename, ConvNetwork.board_to_tensor)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    for round in range(rounds):
        total_loss = 0
        for batch in tqdm(train_loader):
            loss = process_batch(BNet, batch, loss_fn)
            total_loss += loss.item()

            BNet.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in BNet.model.parameters():
                    param -= learning_rate * param.grad

        test_loss = 0
        for batch in test_loader:
            loss = process_batch(BNet, batch, loss_fn)
            test_loss += loss.item()

        print(
            f"Round: {round}\tAvg. Training Loss: {total_loss / len(train_loader):.4f}\t\tAvg. Test Loss: {test_loss / len(test_loader):.4f}"
        )

        torch.save(BNet.model, output_filename)


if __name__ == "__main__":
    main()
