import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("./data/house_prices_dataset.csv")

X = df["Bedrooms"].to_numpy()
Y = df["Price"].to_numpy()

def cost(w, b):
    return np.sum(((X*w + b)-Y)**2)

def gradient(w, b):
    # Derivative of the Cost function
    # Partial sums
    dw = np.sum(2 * X * (X*w + b - Y))
    db = np.sum(2*(X*w + b - Y))
    return dw, db

def predict(x, w, b):
    return w*x + b

def gradient_descent(init_w, init_b, learning_rate = 0.00001, eps = 1e-5):
    cur_w = init_w
    cur_b = init_b

    prev_w = float('inf')
    prev_b = float('inf')

    while abs(cur_w-prev_w) > eps and abs(cur_b-prev_b) > eps:
        prev_w = cur_w
        prev_b = cur_b

        dw, db = gradient(cur_w, cur_b)
        cur_w -= dw*learning_rate
        cur_b -= db*learning_rate
    return cur_w, cur_b

w, b = gradient_descent(6, 9);

for i in range(len(X)):
    print(predict(X[i], w, b), Y[i])

plt.figure(figsize=(8, 5))
plt.scatter(X, Y, alpha=0.6, label="Data points")
x_vals = np.linspace(X.min(), X.max(), 100)
y_vals = predict(x_vals, w, b)
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Gradient Descent Fit")  # Regression line
plt.xlabel("Number of Bedrooms")
plt.ylabel("House Price ($)")
plt.title("House Prices vs. Number of Bedrooms (Gradient Descent)")
plt.legend()
plt.grid(True)
plt.show()
