import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


def data():
    np.random.seed(1)
    X, Y = load_planar_dataset()
    # print(x.shape, y.shape)
    # plt.scatter(x[0, :], x[1, :], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    shape_X, shape_Y = X.shape, Y.shape
    m = Y.shape[1]
    return X, Y, m


def LR():
    """先用逻辑回归进行分类，看看效果"""
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(X.T, Y.T)

    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")
    plt.show()


def layer_size(X, Y):
    n_x, n_y = X.shape[0], Y.shape[0]
    n_h = 4  # 隐藏层含四个节点
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.rand(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    z1 = np.dot(W1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)

    assert (a2.shape == (1, X.shape[1]))

    cache = {"z1": z1,
             "a1": a1,
             "z2": z2,
             "a2": a2}
    return a2, cache


def compute_cost(a2, Y, parameters):
    W1, W2 = parameters['W1'], parameters['W2']
    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = -np.sum(logprobs) / m
    cost = np.squeeze(cost)
    assert (isinstance(cost, float))
    return cost


def back_propagation(parameters, cache, X, Y):
    W1, W2 = parameters['W1'], parameters['W2']
    a1, a2 = cache['a1'], cache['a2']
    dz2 = a2 - Y
    dW2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(W2.T, dz2), 1 - np.power(a1, 2))
    dW1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, lr=1.2):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, iterations=100, print_cost=False):
    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    # W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']

    for i in range(0, iterations):
        a2, cache = forward_propagation(X, parameters)
        cost = compute_cost(a2, Y, parameters)
        grads = back_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    a2, cache = forward_propagation(X, parameters)
    predictions = np.round(a2)
    return predictions


def show_hidden_layer():
    plt.figure(figsize=(8, 16))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, iterations=5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    plt.show()


def on_other_dataset():
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    dataset = "noisy_moons"

    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    # make blobs binary
    if dataset == "blobs":
        Y = Y % 2

    # 可视化
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()


if __name__ == '__main__':
    X, Y, m = data()
    # LR()
    """逐步分析"""
    n_x, n_h, n_y = layer_size(X, Y)  # 设置每一层的节点数
    parameters = initialize_parameters(n_x, n_h, n_y)  # 初始化节点值
    a2, cache = forward_propagation(X, parameters)  # 前向传播
    cost = compute_cost(a2, Y, parameters)  # 计算损失
    grads = back_propagation(parameters, cache, X, Y)  # 反向传播
    parameters = update_parameters(parameters, grads, lr=1.2)  # 更新参数

    """正片开始"""
    parameters = nn_model(X, Y, 4, 10000, False)

    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()

    # 准确率
    predictions = predict(parameters, X)
    print('Accuracy: %d' % float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    # show_hidden_layer()   # 更改隐藏层节点数，测试准确率
    on_other_dataset()  # 在其他数据集上进行测试
