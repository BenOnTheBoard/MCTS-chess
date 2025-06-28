import chess
import torch
from math import tanh
from numpy import float16
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from engine.heuristics.dualHeadNetwork import DualHeadNetwork


class PolicyValueDataset(Dataset):
    def __init__(self, filename, board_conversion, move_conversion):
        self.board_conversion = board_conversion
        self.move_conversion = move_conversion
        self.data = []
        with open(filename, "r") as file:
            for line in tqdm(file.readlines(), desc="Initial data processing"):
                fen, move, value_str = line.strip().split(",")
                value = float16(tanh(int(value_str)))
                self.data.append((fen, value, move))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, value, move_str = self.data[idx]
        line_state = chess.Board(fen=fen)
        move = chess.Move.from_uci(move_str)

        board_tsr = self.board_conversion(line_state)
        value = torch.tensor(value, dtype=torch.float32)
        move_tsr = torch.tensor(self.move_conversion(move), dtype=torch.long)

        return board_tsr, value, move_tsr


def process_batch(network, batch, value_loss_fn, policy_loss_fn):
    board_tensor, target_values, target_moves = batch
    batch_size = len(board_tensor)

    board_tensor = board_tensor.to(device="xpu", dtype=torch.float32)
    board_tensor = board_tensor.view(batch_size, 11, 8, 8)
    target_values = target_values.to("xpu")
    target_moves = target_moves.to("xpu")

    predicted_values, predicted_moves = network.model(board_tensor)
    predicted_moves = predicted_moves.view(batch_size, 4672)
    predicted_values = predicted_values.view(-1)

    value_loss = value_loss_fn(predicted_values, target_values)
    policy_loss = policy_loss_fn(predicted_moves, target_moves)

    return value_loss.to("cpu"), policy_loss.to("cpu")


def main():
    model = None
    network_type = DualHeadNetwork
    network = network_type(model=model)
    network.model = network.model.to("xpu")

    data_filename = "data/DHRand.txt"
    tests_filename = "data/DHTest.txt"
    output_filename = "models/new_dhn.pt"
    value_loss_fn = torch.nn.MSELoss(reduction="sum")
    policy_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_epochs = 100
    init_learning_rate = 1e-4
    batch_size = 128

    dataset = PolicyValueDataset(
        data_filename, network_type.board_to_tensor, network_type.move_to_flat_index
    )
    testset = PolicyValueDataset(
        tests_filename, network_type.board_to_tensor, network_type.move_to_flat_index
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(network.model.parameters(), lr=init_learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(total_epochs):
        network.model.train()
        total_value_loss = 0
        total_policy_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            value_loss, policy_loss = process_batch(
                network, batch, value_loss_fn, policy_loss_fn
            )
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            composite_loss = value_loss + policy_loss
            composite_loss.backward()
            optimizer.step()

        scheduler.step()

        print(f"""
                Epoch: {epoch}
                Learning Rate: {scheduler.get_last_lr()[0]:.6f}
                Value Head Loss: {total_value_loss / len(dataset)}
                Policy Head Loss: {total_policy_loss / len(dataset)}
        """)

        network.model.eval()
        total_value_loss = 0
        total_policy_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                value_loss, policy_loss = process_batch(
                    network, batch, value_loss_fn, policy_loss_fn
                )
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()

        print(f"""
                Value Head Test Loss: {total_value_loss / len(dataset)}
                Policy Head Test Loss: {total_policy_loss / len(dataset)}
        """)
        torch.save(network.model, output_filename)

    print("\n\nTraining is done!")


if __name__ == "__main__":
    main()
