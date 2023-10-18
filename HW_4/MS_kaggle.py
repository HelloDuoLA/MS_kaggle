# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

predictList = []


def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]

    ###### You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]), data[:, 0:6]], axis=1)  # 3:6 3:5 3:4 4:5
    ###### You may modify this section to change the model

    Y = None
    if "Train" in addr:
        Y = np.expand_dims(data[:, 6], axis=1)

    return (X, Y, n)


def cost_gradient(W, X, Y, n):
    Y_head = 1 / (1 + np.exp(-np.dot(X, W)))
    G = (np.sum((Y_head - Y) * X, axis=0) / n).reshape(X.shape[1], 1)  ###### Gradient
    j = np.sum(Y * np.log(1 + np.exp(-X @ W)) + (1 - Y_head) * np.log(
        1 + np.exp(X @ W))) / n  ###### cost with respect to current W

    return (j, G)


def train(W, X, Y, lr, n, iterations):
    ###### You may modify this section to do 10-fold validation
    J = np.zeros([iterations, 1])
    E_trn = np.zeros([iterations, 1])
    E_val = np.zeros([iterations, 1])
    n = int(0.9 * n)
    X_trn = X[:n]
    Y_trn = Y[:n]
    X_val = X[n:]
    Y_val = Y[n:]

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X_trn, Y_trn, n)
        W = W - lr * G
        E_trn[i] = error(W, X_trn, Y_trn)
        E_val[i] = error(W, X_val, Y_val)
    print(E_val[-1])
    ###### You may modify this section to do 10-fold validation

    return (W, J, E_trn, E_val)


def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1

    return (1 - np.mean(np.equal(Y_hat, Y)))


def predict(W, i):
    (X, _, _) = read_data("test_Data.csv")
    Y_hat = 1 / (1 + np.exp(-np.dot(X, W)))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1
    predictList.append(Y_hat)
    idx = np.expand_dims(np.arange(1, 201), axis=1)
    # np.savetxt("predict"+str(i)+".csv", np.concatenate([idx, Y_hat], axis=1), header="Index,ID", comments='', delimiter=',')


iterations = 10000  ###### Training loops
lr = 0.005  ###### Learning rate

(X, Y, n) = read_data("train.csv")
tX = X
tY = Y
for i in range(10):
    W = np.random.random([X.shape[1], 1])
    n = int(0.9 * n)
    tX1 = tX[n:]
    tX2 = tX[:n]
    tY1 = tY[n:]
    tY2 = tY[:n]
    tX = np.concatenate((tX1, tX2), axis=0)
    tY = np.concatenate((tY1, tY2), axis=0)
    (W, J, E_trn, E_val) = train(W, tX, tY, lr, n, iterations)
    predict(W, i)

predictList = np.array(predictList)
predictList = np.sum(predictList, axis=0) / 10
# print(predictList)
predictList[predictList >= 0.5] = 1
predictList[predictList < 0.5] = 0
idx = np.expand_dims(np.arange(1, 201), axis=1)
np.savetxt("predict.csv", np.concatenate([idx, predictList], axis=1), header="Index,ID", comments='', delimiter=',')
###### You may modify this section to do 10-fold validation
# plt.figure()
# plt.plot(range(iterations), J)
# plt.figure()
# plt.ylim(0,1)
# plt.plot(range(iterations), E_trn, "b")
# plt.plot(range(iterations), E_val, "r")
# plt.show()
###### You may modify this section to do 10-fold validation