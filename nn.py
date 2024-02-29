import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class Linear:
    def __init__(self, n_input, units) -> None:
        self.w = np.random.randn(n_input, units) * np.sqrt(2.0 / n_input)
        self.b = np.zeros((1, units))  # Bias initialization.

    def forward(self, x) -> np.ndarray:
        return x @ self.w + self.b


class ReLU:
    def forward(self, x) -> np.ndarray:
        return np.maximum(0, x)


class Softmax:
    def forward(self, x) -> np.ndarray:
        e_z = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)


class NLLL:
    def __init__(self, epsilon=1e-5) -> None:
        self.epsilon = epsilon

    def compute(self, y_pred, y_true) -> np.ndarray:

        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        confidence = y_pred[range(len(y_true)), y_true]

        return np.mean(-np.log(confidence))


def decay(current_iteration, lr, decay=1e-5) -> float:
    return lr * (1.0 / (1 + decay * current_iteration))


if __name__ == "__main__":
    seed = 2025

    # Load MNIST data.
    data = pd.read_csv("mnist_train.csv").to_numpy()

    # Prepare data.
    X = data[:, 1:] / 255.0
    y = data[:, 0]

    n_data, n_features = X.shape

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed
    )

    # Declare architecture.
    units = 128
    l1 = Linear(n_features, units)
    relu1 = ReLU()
    l2 = Linear(units, len(np.unique(y)))
    softmax = Softmax()
    nlll = NLLL()

    # Declare hyper-parameters.
    lr = 1e-2
    decay_rate = 1e-5
    current_lr = lr
    batch_size = 32
    epochs = 30
    batch_shuffle = True

    train_iter_max = X_train.shape[0] // batch_size
    test_iter_max = X_test.shape[0] // batch_size

    # Training.
    print("Training...")
    for epoch in range(1, epochs + 1):

        current_iter = 0

        if batch_shuffle:
            X_train, y_train = shuffle(X_train, y_train, random_state=seed)

        total_loss = 0.0
        total_acc = 0.0
        for iter in range(train_iter_max):

            # Get mini batch.
            batch_x = X_train[iter : iter + batch_size]
            batch_y = y_train[iter : iter + batch_size]

            # Forward propagation.
            l1_out = l1.forward(batch_x)
            relu_out = relu1.forward(l1_out)
            l2_out = l2.forward(relu_out)

            # Compute probabilities.
            softmax_output = softmax.forward(l2_out)

            # Loss Computation.
            loss = nlll.compute(softmax_output, batch_y)
            total_loss += loss

            # Metric Computation.
            predictions = np.argmax(softmax_output, axis=1)
            accuracy = np.mean(predictions == batch_y)
            total_acc += accuracy

            # Compute gradient of the loss w.r.t softmax's output (S).
            dl = softmax_output.copy()
            dl[range(len(softmax_output)), batch_y] -= 1.0
            dl /= dl.shape[0]

            # Compute L2 gradient w.r.t w2.
            dw_l2 = relu_out.T @ dl
            db_l2 = np.sum(dl, axis=0, keepdims=True)

            # Compute ReLU Gradients w.r.t loss (intermediare computation to compute dw_l1).
            drelu = (dl @ l2.w.T) * (relu_out > 0)
            # Compute L1 Gradients.
            dw_l1 = batch_x.T @ drelu
            db_l1 = np.sum(drelu, axis=0, keepdims=True)

            # Update parameters.
            l2.w -= current_lr * dw_l2
            l1.w -= current_lr * dw_l1

            l2.b -= current_lr * db_l2
            l1.b -= current_lr * db_l1

            current_lr = decay(epoch * iter, lr, decay=decay_rate)

        print(
            f"epoch: {epoch} , "
            + f"acc: {total_acc/train_iter_max:.3} , "
            + f"loss: {total_loss/train_iter_max:.3} "
            + f"lr:{current_lr:.3} "
        )

    # Testing.
    print("\nTesting:")
    total_acc = 0.0
    for iter in range(test_iter_max):

        # Get data.
        batch_x = X_test[iter : iter + batch_size]
        batch_y = y_test[iter : iter + batch_size]

        # Forward propagation.
        l1_out = l1.forward(batch_x)
        relu_out = relu1.forward(l1_out)
        l2_out = l2.forward(relu_out)

        # Compute probabilities.
        softmax_output = softmax.forward(l2_out)

        # Metric Computation.
        predictions = np.argmax(softmax_output, axis=1)
        accuracy = np.mean(predictions == batch_y)
        total_acc += accuracy

    print(f"Test accuracy: {total_acc/test_iter_max:.3}")

    # Drawing output.
    n_rows = 0
    n_image_line = 8
    if len(batch_y) == n_image_line:
        fig, ax = plt.subplots(1, n_image_line)
    else:
        n_rows = len(batch_y) // n_image_line
        fig, ax = plt.subplots(n_rows, n_image_line)

    current_row = 0
    for i in range(len(batch_y)):
        prediction_class = np.argmax(softmax_output, axis=1)[i]

        if n_rows > 0:
            ax_i = ax[current_row % n_rows, i % n_image_line]
        else:
            ax_i = ax[i]

        ax_i.imshow(batch_x[i].reshape(28, 28), cmap="gray")
        ax_i.axis("off")  # This turns off the axis lines and labels
        ax_i.set_title(f"{prediction_class}", fontsize=10)

        if (i + 1) % n_image_line == 0:
            current_row += 1

    plt.tight_layout()
    fig.savefig("assets/predictions.png")
