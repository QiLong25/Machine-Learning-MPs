import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

def linear_gd(X,Y,lrate=0.1,num_iter=1000):
    Y = np.reshape(Y, [-1, 1])
    n = np.shape(X)[0]
    d = np.shape(X)[1]
    w = np.zeros((d+1, 1))
    add = np.ones((n, 1))
    X = np.c_[add, X]

    for i in range(num_iter):
        w += -lrate * (X.T @ X @ w - X.T @ Y) / n

    # return parameters as numpy array
    return w

def linear_normal(X,Y):
    Y = np.reshape(Y, [-1, 1])
    n = np.shape(X)[0]
    add = np.ones((n, 1))
    X = np.c_[add, X]

    temp = X.T @ X
    w = (np.linalg.inv(temp)) @ X.T @ Y

    # return parameters as numpy array
    return w

def plot_linear():
    X, Y = utils.load_reg_data()
    Y = np.reshape(Y, [-1, 1])
    n = np.shape(X)[0]
    add = np.ones((n, 1))
    X2 = np.c_[add, X]

    w = linear_normal(X, Y)
    result = plt.figure()
    plt.scatter(X, Y)
    print(list(X.T[0]), list((X2 @ w).T[0]))
    plt.plot(list(X.T[0]), list((X2 @ w).T[0]))
    plt.show()

    return result

print(plot_linear())
