from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def create_dataset() -> Tuple[np.array, np.array]:
    # Creating the dataset
    samples = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    labels = np.array([-1, 1, 1, -1])
    return samples, labels


def init_model() -> Tuple[np.array, float, np.array, float]:
    # Random initializing parameters weights
    np.random.seed(101)
    w_init = np.random.rand(2)
    b2_init = np.random.rand(1)
    u_init = np.random.rand(2, 2)
    b1_init = np.random.rand(1)
    return w_init, b2_init, u_init, b1_init


def main():
    # Creating the dataset
    x, y = create_dataset()

    # Random initialization of the model's parameters
    w, b2, u, b1 = init_model()

    # Training the model
    num_epochs = 100
    learning_rate = 0.001
    loss = []
    for e in range(num_epochs):
        # Calculating the model's output
        h = np.maximum(np.matmul(u, x.T) + b1, 0)
        f = np.matmul(w, h) + b2

        # Calculating the optimization criteria
        epoch_loss = np.sum((y - f) ** 2)
        loss.append(epoch_loss)

        # Calculating the derivatives
        dl_dw = np.sum(-2 * np.matmul(h, (y - f)))
        dl_db2 = np.sum(-2 * np.matmul(np.ones_like(y), (y - f)))
        dl_du = np.sum(-2 * np.matmul((y - f), np.matmul(w, x.T)))
        dl_db1 = np.sum(-2 * np.matmul((y - f), np.matmul(w, np.ones_like(x.T))))

        # Applying Gradient Descent to optimize the model
        w = w - learning_rate * dl_dw
        b2 = b2 - learning_rate * dl_db2
        u = u - learning_rate * dl_du
        b1 = b1 - learning_rate * dl_db1

    plt.figure()
    plt.plot(loss)
    plt.title('Model Optimization\n' + r'$\mathcal{L} = (y - f(x)) ^2$')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    print('Final Model parameters:')
    print(f'\tW = {w}')
    print(f'\tb2 = {b2}')
    print(f'\tU = {u}')
    print(f'\tb1 = {b1}')


if __name__ == "__main__":
    main()
