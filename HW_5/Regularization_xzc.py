# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]
    
    ###### You may modify this section to change the model
    x1 = data[:,[0,1,2,3,4,5,6,7]]
    X = np.ones([n, 1])
    for i in range(poly_degree):
        X = np.concatenate([X,x1**(i + 1)], axis=1)
    # X = np.concatenate([np.ones([n, 1]),
    #                     np.expand_dims(np.power(data[:,0],1), axis=1),
    #                     np.expand_dims(np.power(data[:,1],2), axis=1),
    #                     np.expand_dims(np.power(data[:,2],3), axis=1),
    #                     np.expand_dims(np.power(data[:,3],4), axis=1),
    #                     np.expand_dims(np.power(data[:,4],1), axis=1),
    #                     np.expand_dims(np.power(data[:,5],2), axis=1),
    #                     np.expand_dims(np.power(data[:,6],3), axis=1),
    #                     np.expand_dims(np.power(data[:,7],4), axis=1)],
    #                     axis=1)
    ###### You may modify this section to change the model

    Y = None
    if "Train" in addr:
        Y = np.expand_dims(data[:, -1], axis=1)
    
    return (X,Y,n)

def cost_gradient(W, X, Y, n, lambd):
    Y_head = 1 / (1 + np.exp(-np.dot(X,W)))
    if (Regula_type == "l2"):
        j = np.sum(Y * np.log(1 + np.exp(-X @ W)) + (1 - Y_head) * np.log(1 + np.exp(X @ W))) / n +  lambd / 2 * sum(W**2) 
        G = (np.sum((Y_head - Y) * X ,axis = 0) / n).reshape(X.shape[1],1) + lambd * W  
    elif (Regula_type == "l1"):
        j = np.sum(Y * np.log(1 + np.exp(-X @ W)) + (1 - Y_head) * np.log(1 + np.exp(X @ W))) / n +  lambd  * sum(abs(W)) 
        G = (np.sum((Y_head - Y) * X ,axis = 0) / n).reshape(X.shape[1],1) + lambd * np.sign(W)  
    
    return (j, G)

def train(W, X, Y, lr, n, iterations, lambd):
    J = np.zeros([iterations, 1])
    
    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n, lambd)
        W = W - lr*G
        err = error(W, X, Y)
        print("iteration %d err : %f"%(i,err))
    return (W,J,err)

def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    return (1-np.mean(np.equal(Y_hat, Y)))

def show_W(w):
    print(" "*7 + "w0" + " "*7,end="")
    print("w1" + " "*8,end="")
    print("w2" + " "*8,end="")
    print("w3" + " "*8,end="")
    print("w4" + " "*8,end="")
    print("w5" + " "*8,end="")
    print("w6" + " "*8,end="")
    print("w7" + " "*8,end="")
    print()

    for i in range(poly_degree):
        print(" %d "%(i),end="")
        for j in range(8):
            print('%+07f '%(w[i * 8 + 1 + j]),end="")
        print("")

    print("b : %+07f"%(W[0]))

def predict(W):
    (X, _, _) = read_data("Test_Data.csv")
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    idx = np.expand_dims(np.arange(1,201), axis=1)
    np.savetxt("predict_xzc.csv", np.concatenate([idx, Y_hat], axis=1), header = "Index,ID", comments='', delimiter=',')
    
iterations = 50000 ###### Training loops
lr = 0.005          ###### Learning rate
lambd =  0.01 ##### Lambda to control the weight of Regularization part
poly_degree = 4
Regula_type = "l2" 
# Regula_type = "l1" 


(X, Y, n) = read_data("Train.csv")
W = np.random.random([X.shape[1], 1])
(W,J,err) = train(W, X, Y, lr, n, iterations, lambd)

# print(W)
# print(W[0])
show_W(W)

plt.figure()
# plt.plot(range(iterations), J)

predict(W)