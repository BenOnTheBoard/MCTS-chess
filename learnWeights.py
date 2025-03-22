import chess
import torch

from engine.heuristics.basicNetwork import BasicNetwork


def main():
    BNet = BasicNetwork()

    data_filenames = (
        "training_data.txt",
        "training_data_rand.txt",
    )
    tests_filename = "testing_data.txt"
    output_filename = "saved_model.pt"
    loss_fn = torch.nn.MSELoss()
    rounds = 20
    learning_rate = 1e-5

    dataset = []
    for dfn in data_filenames:
        with open(dfn, "r") as data_file:
            dataset.extend(data_file.readlines())

    with open(tests_filename, "r") as test_data_file:
        testset = test_data_file.readlines()

    for round in range(rounds):
        total_loss = 0
        for line in dataset:
            fen, y = line.strip().split(",")
            l_state = chess.Board(fen=fen)

            y = torch.as_tensor(
                [int(y)],
                dtype=torch.float,
            )
            y = torch.sigmoid(y)
            y_pred = BNet.tensor_eval(l_state)

            loss = loss_fn(y_pred, y)
            total_loss += loss

            BNet.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in BNet.model.parameters():
                    param -= learning_rate * param.grad

        test_loss = 0
        for line in testset:
            fen, y = line.strip().split(",")
            l_state = chess.Board(fen=fen)

            y = torch.as_tensor(
                [int(y)],
                dtype=torch.float,
            )
            y = torch.sigmoid(y)
            y_pred = BNet.tensor_eval(l_state)

            loss = loss_fn(y_pred, y)
            test_loss += loss

        print(
            f"Round: {round}\tAvg. Training Loss: {total_loss / len(dataset):.4f}\t\tAvg. Test Loss: {test_loss / len(testset):.4f}"
        )

    torch.save(BNet.model, output_filename)


if __name__ == "__main__":
    main()
