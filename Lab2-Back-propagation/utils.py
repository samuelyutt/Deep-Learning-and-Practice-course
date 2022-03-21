import numpy as np


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