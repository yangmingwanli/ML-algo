import numpy as np
import matplotlib.pyplot as plt
# linear regression y = 2 * x1 + 3 * x2 + 5
# create X array (x1,x2) 100x2 array with random number between 1 and 10
x = np.random.randint(1,10,(100,2))
# add constant column
x = np.append(x, np.ones((100,1)), axis = 1)
# get y
y = x.dot(np.array([2,3,5]))
# add noise
y += np.random.randn(100)
# create a random starting values for theta
theta = np.random.randn(3)

def gradient_descent(x, y, theta, lr):
    # J is loss function, dJ is dJ/dtheta, lr is learning rate
    J = np.transpose(x.dot(theta) - y).dot(x.dot(theta) - y)
    dJ = np.transpose(x).dot(x.dot(theta) - y)
    theta -= dJ * lr
    return J, dJ, theta

lr = 1e-5
J_his = []
for _ in range(1000000):
    J, _, theta = gradient_descent(x, y, theta, lr)
    J_his.append(J)
print('theta found with GD is', theta)
# solve it with normal equation
theta = np.linalg.pinv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
print('theta found with normal equation is', theta)
# plot the loss function
plt.plot(J_his)
plt.show()

# Math
# X is nxm dimension array, n is number of samples, m is number of features
# Y is nx1 dimension array
# Theta is mx1 dimension array
# J = (X * Theta - Y)T * (X * Theta - Y)
# J = (Theta)T * (X)T * X * Theta - (Theta)T * (X)T * Y - (Y)T * X * Theta - (Y)T * Y
# dJ/dTheta = 2 * (X)T * X * Theta - 2 * (X)T * Y = 2 * (X)T * (X * Theta - Y)
# normal equation set dJ/dTheta = 0  =>   Theta = inv((X)T * X) * ((X)T * Y)
