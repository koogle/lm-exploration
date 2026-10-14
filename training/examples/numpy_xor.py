import numpy as np


def create_examples(num_entries):
    input1 = np.random.randn(num_entries) > 0
    input2 = np.logical_not(input1)
    input2[2::3] = input1[2::3]
    labels = np.logical_xor(input1, input2)

    A, B, Y = input1, input2, labels
    return A, B, Y


class Linear:
    def __init__(self, n_in, n_out):
        self.weights = np.random.randn(n_out, n_in + 1)

    def __call__(self, x):
        x = np.asarray(x)
        ones = np.ones(x.shape[:-1] + (1,))
        x = np.concatenate((x, ones), axis=-1)
        self.input = x
        return x @ self.weights.T

    def backprop(self, gradient):
        gradient = np.asarray(gradient)
        self.weights_grad = gradient.T @ self.input
        return gradient @ self.weights[:, :-1]


def loss(target, result):
    return (np.asarray(target) - np.asarray(result)) ** 2


def loss_derivative(target, result):
    return 2 * (np.asarray(result) - np.asarray(target))


class ReLU:
    def __call__(self, x):
        self.input = np.asarray(x)
        return np.maximum(0, x)

    def backprop(self, gradient):
        return np.asarray(gradient) * (self.input > 0)


if __name__ == "__main__":
    np.random.seed(0)
    A, B, Y = create_examples(20)
    X = np.column_stack((A, B)).astype(float)
    Y = Y.astype(float).reshape(-1, 1)
    linear1 = Linear(n_in=2, n_out=5)
    relu = ReLU()
    linear2 = Linear(n_in=5, n_out=1)
    batch_size = 5
    learning_rate = 0.01
    epochs = 1000

    outputs = linear2(relu(linear1(X)))
    print(f"Epoch {0:4d}: loss = {loss(Y, outputs).mean():.8f}")

    for epoch in range(1, epochs + 1):
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            targets = Y[start : start + batch_size]
            outputs = linear2(relu(linear1(batch)))

            gradient = loss_derivative(targets, outputs) / outputs.size
            gradient = linear2.backprop(gradient)
            gradient = relu.backprop(gradient)
            linear1.backprop(gradient)

            linear1.weights -= learning_rate * linear1.weights_grad
            linear2.weights -= learning_rate * linear2.weights_grad

        if epoch % 100 == 0:
            outputs = linear2(relu(linear1(X)))
            print(f"Epoch {epoch:4d}: loss = {loss(Y, outputs).mean():.8f}")
