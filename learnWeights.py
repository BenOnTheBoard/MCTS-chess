from math import log10

import chess
import torch

from engine.heuristics.basicNetwork import BasicNetwork


def main():
    BNet = BasicNetwork()

    data_filename = "training_data.txt"
    tests_filename = "testing_data.txt"
    output_filename = "saved_model.pt"
    loss_fn = torch.nn.MSELoss()
    rounds = 10
    learning_rate = 1e-5

    with open(data_filename, "r") as data_file:
        dataset = data_file.readlines()

    for round in range(rounds):
        total_loss = 0
        for line in dataset:
            fen, y = line.strip().split(",")
            l_state = chess.Board(fen=fen)

            y = torch.as_tensor(
                [int(y)],
                dtype=torch.float,
            )
            y_pred = BNet.tensor_eval(l_state)

            loss = loss_fn(y_pred, y)
            total_loss += loss

            BNet.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in BNet.model.parameters():
                    param -= learning_rate * param.grad
        print(f"Round: {round + 1},\tAverage Loss: {total_loss / len(dataset)}")

    with open(tests_filename, "r") as test_data_file:
        testset = test_data_file.readlines()

    total_loss = 0
    for line in testset:
        fen, y = line.strip().split(",")
        l_state = chess.Board(fen=fen)

        y = torch.as_tensor(
            [int(y)],
            dtype=torch.float,
        )
        y_pred = BNet.tensor_eval(l_state)

        loss = loss_fn(y_pred, y)
        total_loss += loss

    test_result = total_loss / len(testset)
    print(f"\n\nTest, Average Loss:\t{test_result} = 10 ^ {log10(test_result)}")
    print(BNet.evaluate(chess.Board()))

    torch.save(BNet.model, output_filename)


if __name__ == "__main__":
    main()
