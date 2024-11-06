import matplotlib.pyplot as plt
import numpy as np

'''
Let's take notes
First, we will try on a very simple data set, consists only from one attribute.
I think we will try this:
[[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 11]]
data = [
    {'x': 1, 'y': 2},
    {'x': 2, 'y': 4},
    {'x': 4, 'y': 8},
    {'x': 6, 'y': 12},
    {'x': 7, 'y': 14},
    {'x': 8, 'y': 16},
    {'x': 10, 'y': 20},
    {'x': 12, 'y': 24},
]

Then, what we are trying to do is to find the equation --> y = w*x (we will add +b later) using linear regression
'''
w = 23 # weight (we will use to predict) NOTE: 23 is just random initialization, you can initilize it with anything.

def cost(w):
    res = 0
    for obj in data:
        res += pow(obj['x']*w - obj['y'], 2)
    return res

def predict(x, w):
    return x*w

data = [
    {'x': 1, 'y': 3},
    {'x': 2, 'y': 6},
    {'x': 4, 'y': 12},
    {'x': 5, 'y': 13},
    {'x': 6, 'y': 18},
    {'x': 7, 'y': 21},
    {'x': 8, 'y': 24},
    {'x': 9, 'y': 26},
    {'x': 10, 'y': 30},
    {'x': 12, 'y': 36},
    {'x': 13, 'y': 35.8},
]

def train(w):
    '''
    Train the model using Batch Gradient Descnet
    '''
    new_w = w
    iterations = 10000
    alpha = 0.0001

    for g in range(iterations):
        sum = 0
        for obj in data:
            sum += obj['y'] - obj['x']*new_w
        new_w = new_w + alpha * sum
    
    return new_w

w = train(w)

print("w = ", w)
print("cost(w) = ", cost(w))
print("predict 100: ", predict(100, w))

X = [point['x'] for point in data]
Y = [point['y'] for point in data]

PREDICT_Y = [w*point['x'] for point in data]

plt.scatter(X, Y, color='blue')
plt.plot(X, PREDICT_Y, label=f'y = {w} * x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = w * x')
plt.legend()

plt.grid(True)
plt.show()

