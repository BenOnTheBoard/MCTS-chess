import chess
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine.heuristics.simpleResNetPolicy import SimpleResNetPolicy


class PolicyDataset(Dataset):
    def __init__(self, filename, board_conversion, move_conversion):
        self.board_conversion = board_conversion
        self.move_conversion = move_conversion
        self.data = []
        with open(filename, "r") as file:
            self.data = file.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move = self.data[idx].strip().split(",")
        line_state = chess.Board(fen=fen)
        line_tsr = self.board_conversion(line_state)

        y = self.move_conversion(chess.Move.from_uci(move))

        return line_tsr, torch.tensor(y, dtype=torch.long)


def process_batch(network, batch, loss_fn):
    board_tensor, targets = batch
    batch_size = len(board_tensor)

    board_tensor = board_tensor.to(torch.float32)
    board_tensor = board_tensor.view(batch_size, 11, 8, 8)

    predictions = network.model(board_tensor).view(batch_size, 4672)
    loss = loss_fn(predictions, targets)

    return loss


def main():
    model = None
    network_type = SimpleResNetPolicy
    network = network_type(model=model)

    data_filename = "data/MoveRand.txt"
    tests_filename = "data/MoveTest.txt"
    output_filename = "models/new_srnp.pt"
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_epochs = 100
    init_learning_rate = 1e-2
    batch_size = 32

    dataset = PolicyDataset(
        data_filename, network_type.board_to_tensor, network_type.move_to_flat_index
    )
    testset = PolicyDataset(
        tests_filename, network_type.board_to_tensor, network_type.move_to_flat_index
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(network.model.parameters(), lr=init_learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(total_epochs):
        network.model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            loss = process_batch(network, batch, loss_fn)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f"""
                Epoch: {epoch}
                Learning Rate: {scheduler.get_last_lr()[0]:.6f}
                Training Loss: {total_loss / len(dataset)}
        """)

        network.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                loss = process_batch(network, batch, loss_fn)
                test_loss += loss.item()

        print(f"""
                Model test loss: {test_loss / len(testset):.4f}
        """)
        torch.save(network.model, output_filename)

    print("\n\nTraining is done!")


if __name__ == "__main__":
    main()
