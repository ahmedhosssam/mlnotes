import numpy as np
from math import *
'''
Gradient descent is an optimization algorithm that iteratively updates parameters in the direction of the negative gradient to minimize a loss function.
'''

def f(X):
    '''
    X: numpy array
    '''
    return sin(X[0]) + (X[1]**2 * e**X[2]) + log(X[3] + 2) + 1 / (X[4] + 3)

def gradient(X):
    '''
    X: numpy array
    '''
    res = np.zeros(len(X))
    res[0] = cos(X[0])
    res[1] = 2*X[1] * e**X[2]
    res[2] = X[1]**2 * e**X[2]
    res[3] = 1 / (X[3] + 2)
    res[4] = -1 / (X[4] + 3)**2
    return res

def gradient_descent(init_d, learning_rate = 0.0001):
    cur = np.copy(init_d)
    prev = np.full(len(init_d), np.inf)
    eps = np.full(len(init_d), 1e-5)

    for i in range(1000):
        prev = np.copy(cur)
        grad = gradient(cur)
        cur = cur - grad*learning_rate
    return cur

arr = np.array([5, 5, 5, 5, 5])
print(arr)
print(gradient_descent(arr))
