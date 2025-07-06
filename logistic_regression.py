"""
If we have some input data:
x1, x2, y
1,2,1
2,3,1
3,4,0
4,5,0
5,5,1
8,5,0

in order to perform logreg, we will need:
    - cost function (log loss)
    - derivative of log loss
    - apply gradient descent
"""

import numpy as np
import pandas as pd

def log_loss(Y, P):
    eps = 1e-15
    P = np.clip(P, eps, 1 - eps)
    
    return -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))

def deriv_log_loss(X, Y, P, W):
    error = P - Y
    dw = X.T.dot(error) / X.shape[0]
    return dw

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, Y):
    W = np.random.rand(2, 1)
    iter = 1000
    lr = 0.1

    while iter > 0:
        P = sigmoid(X.dot(W))
        cost = log_loss(Y, P)
        d_cost = deriv_log_loss(X, Y, P, W)
        W = W - lr*d_cost
        iter -= 1

    return W

def main():

    # implement gradient descent

    df = pd.read_csv("./data/logreg_data.csv")
    X = df[['x1', 'x2']]
    Y = df['y'].values.reshape(-1, 1)

    W = gradient_descent(X, Y)

    #val_list = [-1.1063349740060282,1.158595579007404]
    #val = np.array(val_list)
    #print(val.dot(W))

    YES = 0
    NO = 0
    for idx, row in df.iterrows():
        x = np.array([row['x1'], row['x2']])
        pred = sigmoid(x.dot(W))[0]
        y = row['y']
        if (y == 1 and pred >= 0.5) or (y == 0 and pred < 0.5):
            YES += 1

        #print(f"y = {y}, pred = {pred}")

    print(W)
    acc = YES/100
    print(acc)

if __name__ == "__main__":
    main()
