import numpy as np
import matplotlib.pyplot as plt


class MyLoss():
    def apply(GT, X):
        return np.sum(np.square(np.subtract(GT, X))) / 2

    def partial_derivative(GT, X, term):
        return -(GT[term] - X[term])


class MSELoss():
    def apply(GT, X):
        return np.sum(np.square(np.subtract(GT, X)))

    def partial_derivative(GT, X, term):
        return -2 * (GT[term] - X[term])


class MAELoss():
    def apply(GT, X):
        return np.sum(np.abs(np.subtract(GT, X)))

    def partial_derivative(GT, X, term):
        if X[term] > GT[term]:
            return 1
        elif X[term] < GT[term]:
            return -1
        else:
            return 0


class Sigmoid():
    def apply(x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(x):
        return np.multiply(x, 1.0 - x)


class NoActivation():
    def apply(x):
        return x

    def derivative(x):
        return 1.0


class ReLu():
    def apply(x):
        if x >= 0:
            return x
        else:
            return 0.0

    def derivative(x):
        if x >= 0:
            return 1.0
        else:
            return 0


class TanH():
    def apply(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(x):
        return 1 - TanH.apply(x)


def show_result(x, y, pred_y, loss, **kwargs):
    suptitle = ', '.join([f'{key}={value}' for key, value in kwargs.items()])
    plt.figure(figsize=(12, 4))
    plt.suptitle(suptitle)

    plt.subplot(1, 3, 1)
    plt.title('Ground truth')
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 3, 2)
    plt.title('Predict result')
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 3, 3)
    plt.title('Learning curve')
    plt.plot(loss)

    plt.savefig(f'../out/{suptitle}.png')
    # plt.show()
