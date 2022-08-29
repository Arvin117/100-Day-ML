import math
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def Lagrangian(w, alpha):
    first_part = np.sum(alpha)
    second_part = np.sum(np.dot(alpha*alpha*y*y*X.T, X))
    res = first_part - 0.5 * second_part
    return res

def gradient(w, X, y, b, lr):
    for i in range(2000):
        for idx, x_i in enumerate(X):
            y_i = y[idx]

            cond = y_i * (np.dot(x_i, w) - b) >= 1
            if cond:
                w -= lr * 2 * w
            else:
                w -= lr * (2 * w - np.dot(x_i, y_i))

    return w, b


def predict(X, w, b):
    pred = np.dot(X, w) - b
    return np.sign(pred)

if __name__ == '__main__':
    np.random.seed(12)
    num_observations = 50

    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
    y = np.where(y <= 0, -1, 1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
    # plt.show()

    w, b, lr = np.random.random(X.shape[1]), 0, 0.001
    w, b = gradient(w, X, y, b, lr)

    pred = predict(X, w, b)
    print(accuracy_score(y, pred))


















