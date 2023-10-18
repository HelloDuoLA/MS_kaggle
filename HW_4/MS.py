# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# 分类问题，二分类问题

def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')
    n = data.shape[0]
    ###### You may modify this section to change the model
    # 改变数字可以改变模型的输入特征
    # 为什么要加一列1？是偏置项
    x1 = data[:,[0,1,2,3,4,5]]
    # x2 = x1 ** 2  #二次项
    # x3 = x1 ** 3  #三次项
    # x4 = x1 ** 4  #四次项
    # x5 = x1 ** 5  #五次项
    # x6 = x1 ** 6  #五次项
    # x7 = x1 ** 7  #五次项
    X = np.ones([n, 1])
    for i in range(poly_degree):
        X = np.concatenate([X,x1**(i + 1)], axis=1)
    
    # X = np.concatenate([np.ones([n, 1]),x1,x2,x3,x4,x5,x6,x6], axis=1)
    ###### You may modify this section to change the model
    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, 6], axis=1)
    return (X,Y,n)

def cost_gradient(W, X, Y, n):
    Y_head = 1 / (1 + np.exp(-np.dot(X,W))) ## TODO:dot和@的区别？？？
    ###### Gradient,梯度
    # TODO:梯度与j(损失有没有什么关系？能不能直接用上)
    G = (np.sum((Y_head - Y) * X ,axis = 0) / n).reshape(X.shape[1],1)  
    ###### cost with respect to current W                 
    j = np.sum(Y * np.log(1 + np.exp(-X @ W)) + (1 - Y_head) * np.log(1 + np.exp(X @ W))) / n  ###### cost with respect to current W
    return (j, G)

def train(W, X, Y, lr, n, iterations):
    ###### You may modify this section to do 10-fold validation
    J = np.zeros([iterations, 1])
    E_trn = np.zeros([iterations, 1])
    E_val = np.zeros([iterations, 1])
    n = int((1 - (10 / nzhe) * 0.1)*n)
    X_trn = X[:n]
    Y_trn = Y[:n]
    X_val = X[n:]
    Y_val = Y[n:]

    # 使用梯度下降算法训练模型
    # 问题:偏置项是怎么更新的？？？
    # 回答：w的第一列是偏置项, X的第一列为1
    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X_trn, Y_trn, n)
        W = W - lr * G
        E_trn[i] = error(W, X_trn, Y_trn)
        E_val[i] = error(W, X_val, Y_val)
        print("iteration %d E_trn : %f, E_val : %f"%(i,E_trn[i],E_val[i]))
    # 打印最后一个迭代的验证误差
    # print("E_val[-1] = ",E_val[-1])
    
    ###### You may modify this section to do 10-fold validation
    return (W,J,E_trn,E_val)

def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    return (1-np.mean(np.equal(Y_hat, Y)))

def predict(W):
    (X, _, _) = read_data("test_Data.csv")
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    idx = np.expand_dims(np.arange(1,201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header="Index,ID", comments='', delimiter=',')

iterations = 200000
lr = 0.005
nzhe = 10
poly_degree = 7

(X, Y, n) = read_data("train.csv")
W = np.random.random([X.shape[1], 1]) + 1
W_mat = [W.copy() for _ in range(nzhe)]
E_trn = np.ones(nzhe)
E_val = np.ones(nzhe)
tX = X
tY = Y
for i in range(nzhe):
    tmpn=int((1 - (10 / nzhe) * 0.1)*n)
    tX1=tX[tmpn:]
    tX2=tX[:tmpn]
    tY1=tY[tmpn:]
    tY2=tY[:tmpn]
    tX=np.concatenate((tX1,tX2),axis=0)
    tY=np.concatenate((tY1,tY2),axis=0)
    (W_mat[i],J,E_trn_tmp,E_tmp) = train(W, tX, tY, lr, n, iterations)
    E_val[i] = E_tmp[-1]
    E_trn[i] = E_trn_tmp[-1]

print("E_val = ",E_val)
print("E_trn = ",E_trn)
W = np.mean(W_mat, axis=0)
# W = W_mat[0]
###### You may modify this section to do 10-fold validation
# plt.figure()
# plt.plot(range(iterations), J)
# plt.figure()
# plt.ylim(0,1)
# plt.plot(range(iterations), E_trn, "b")
# plt.plot(range(iterations), E_val, "r")
###### You may modify this section to do 10-fold validation
predict(W)
