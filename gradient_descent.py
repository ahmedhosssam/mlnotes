'''
Gradient descent is an optimization algorithm that iteratively updates parameters in the direction of the negative gradient to minimize a loss function.
'''

def f(x, y):
    return 4 * (1-x)**2 + (y-5)**2

def gradient(x, y):
    # returns partial derivatives for x and y
    return (-8 + 8*x, 2*y-10)

def gradient_descent(init_x, init_y, learning_rate = 0.001, eps = 1e-5):
    cur_x = init_x
    cur_y = init_y
    prev_x = float('inf')
    prev_y = float('inf')

    while abs(cur_x-prev_x) > eps and abs(cur_y-prev_y) > eps:
        prev_x = cur_x
        prev_y = cur_x
        gx, gy = gradient(cur_x, cur_y)
        cur_x -= gx*learning_rate
        cur_y -= gy*learning_rate
        print(cur_x, cur_y)
    return cur_x, cur_y

x, y = gradient_descent(3, 2)

print(f(x+0.1, y+0.1))  # 0.18205109816324583
print(f(x, y))          # 0.22617791614539254 <-- Minimum
print(f(x-0.1, y-0.1))  # 0.37030473412753917
