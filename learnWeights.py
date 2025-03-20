import chess
import torch

from engine.heuristics.basicNetwork import BasicNetwork


def main():
    BNet = BasicNetwork()

    filename = "training_data.txt"
    loss_fn = torch.nn.MSELoss(reduction="sum")
    rounds = 2000
    learning_rate = 1e-6

    for round in range(rounds):
        total_loss = 0
        with open(filename, "r") as data_file:
            dataset = data_file.readlines()
            for line in dataset:
                fen, y = line.strip().split(",")
                l_state = chess.Board(fen=fen)

                y = torch.as_tensor(int(y), dtype=torch.float)
                y_pred = BNet.evaluate(l_state)

                loss = loss_fn(y_pred, y)
                total_loss += loss

                BNet.model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for param in BNet.model.parameters():
                        param -= learning_rate * param.grad
        print(f"Round: {round}, Total Loss: {total_loss}")


if __name__ == "__main__":
    main()
