import os
import struct
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier  # K-NN


def Iris():
    # 创建数据
    import sklearn.datasets as datasets

    iris = datasets.load_iris()  # 数据：蓝蝴蝶
    X = iris['data']
    Y = iris['target']

    # 训练集和测试集
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0, shuffle=True)

    # 实例化
    Knn = KNeighborsClassifier(n_neighbors=10)
    Knn.fit(X_train, Y_train)
    print('数据组结果:')
    print('Score: ', Knn.score(X_train, Y_train))

    y_pred = Knn.predict(X_test)
    print('Target: ', Y_test)
    print('Result: ', y_pred)


def load_mnist(path, kind):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels



def MyMnist():
    X_train, Y_train = load_mnist('./datasets/MNIST/raw', kind='train')
    X_test, Y_test = load_mnist('./datasets/MNIST/raw', kind='t10k')
    X_train, Y_train, X_test, Y_test = X_train[:10000, :], Y_train[:10000], X_test[:30, :], Y_test[:30]
    # print(X_train.shape)
    print('MNIST数据组结果:')
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    print('Score: ', knn.score(X_train, Y_train))

    y_pred = knn.predict(X_test)
    print('Target: ', Y_test)
    print('Result: ', y_pred)

    '''可视化看一下数据'''
    # fig, ax = plt.subplots(
    #     nrows=2,
    #     ncols=5,
    #     sharex=True,
    #     sharey=True, )
    #
    # ax = ax.flatten()
    # for i in range(10):
    #     img = X_train[Y_train == i][0].reshape(28, 28)
    #     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    #
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    Iris()
    MyMnist()
