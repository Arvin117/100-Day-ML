import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def data():
    x_train, y_train, x_test, y_test, classes = load_dataset()
    print(x_train.shape, y_train.shape)
    # plt.imshow(x_train[25])
    # print("y = " + str(y_train[:, 25]) + ", it's a " + classes[np.squeeze(y_train[:, 25])].decode("utf-8"))
    # plt.show()

    # 将每幅图片展平
    x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
    x_test_flatten = x_test.reshape(x_test.shape[0], -1).T
    print(x_train_flatten.shape, y_train.shape, x_test_flatten.shape, y_test.shape)
    # 标准化
    x_train = x_train_flatten / 255
    x_test = x_test_flatten / 255

    return x_train, y_train, x_test, y_test, classes


# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# print(sigmoid(0))


def initialize_Wb(dim):
    """w.shape = (dim, 1), w = 0, b = 0"""
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, x, y):
    """计算损失值和梯度"""
    m = x.shape[1]
    A = sigmoid(np.dot(w.T, x) + b)  # 输出值
    cost = (-1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))  # 损失函数

    dw = (1 / m) * np.dot(x, (A - y).T)
    db = (1 / m) * np.sum(A - y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost


# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# grads, cost = propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))


def optimize(w, b, x, y, iterations, lr, print_cost=False):
    """反向传播，更新w,b"""
    costs = []
    for i in range(iterations):
        grads, cost = propagate(w, b, x, y)
        dw, db = grads["dw"], grads["db"]
        w = w - lr * dw
        b = b - lr * db
        # 记录cost
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


# params, grads, costs = optimize(w, b, X, Y, iterations= 100, lr = 0.009, print_cost = True)
#
# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))


def predict(w, b, x):
    """预测预测开始预测"""
    m = x.shape[1]
    y_pred = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)
    A = sigmoid(np.dot(w.T, x) + b)
    for i in range(A.shape[1]):
        y_pred[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (y_pred.shape == (1, m))
    return y_pred


# print("predictions = " + str(predict(w, b, X)))


def model(x_train, y_train, x_test, y_test, iterations=200, lr=0.5, print_cost=False):
    w, b = initialize_Wb(x_train.shape[0])
    parameters, grads, costs = optimize(w, b, x_train, y_train, iterations, lr, print_cost)
    w, b = parameters["w"], parameters["b"]

    y_pred_train = predict(w, b, x_train)
    y_pred_test = predict(w, b, x_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": y_pred_test,
         "Y_prediction_train": y_pred_train,
         "w": w,
         "b": b,
         "learning_rate": lr,
         "num_iterations": iterations}

    return d


def draw_cure(d):
    # 绘制学习曲线
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()


def lr_compare(x_train, y_train, x_test, y_test):
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(x_train, y_train, x_test, y_test, iterations=1500, lr=i,
                               print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def test(d, classes, path):
    img = np.array(cv2.imread(path))
    # my_image = np.array(Image.fromarray(img).resize((64, 64)))
    # my_image = my_image.reshape((1, 64 * 64 * 3)).T
    my_image = cv2.resize(img, (64, 64)).reshape((1, 64 * 64 * 3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    cv2.imshow("Image", img)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, classes = data()
    d = model(x_train, y_train, x_test, y_test, iterations=500, lr=0.005, print_cost=True)
    # draw_cure(d)
    # lr_compare(x_train, y_train, x_test, y_test)
    test(d, classes, "datasets/my_img.png")
