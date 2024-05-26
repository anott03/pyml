'''
@auther anott03
a simple linear regression algorithm (that will get less simple over time)
'''

import numpy as np
import matplotlib.pyplot as plt
import math

'''
@param X: 1D matrix of input data
@param y: 1D matrix of expected outputs
@param theta: the slope we are testing
@return J: the cost (error)
'''
def compute_cost(X, y, theta) -> float:
    m = len(y)
    diffs = np.power(predict(X, theta) - y, 2)
    J = 1/(2*m) * np.sum(diffs)
    return J

'''
@param X: input data
@param theta: the slope
@return the set of predctions
'''
def predict(X, theta) -> np.ndarray:
    return np.dot(X, theta)

'''
@param X: 1D matrix of input data
@param y: 1D matrix of expected results
@param initial_theta: the value at which to start theta
@param alpha: how much to increment theta by each iteration
@param iterations: the number of iterations to complete
@return theta
'''
def gradient_descent(X: np.ndarray, y: np.ndarray,
                     inital_theta: float, alpha: float, iterations: int) -> float:
    theta = inital_theta

    m = len(y)
    for _ in range(iterations):
        arr = np.zeros((2, m))

        for i in range(m):
            arr[0, i] = (theta * X[i]) - y[i] # difference between calculated and actual
            arr[1, i] = (theta * X[i]) - y[i] * X[i]

        theta -= alpha/m * sum(arr[0, :])

    return theta

'''
@param X: 1D matrix of input data
@param y: 1D matrix of expected results
@param initial_theta: the value at which to start theta
@param alpha: how much to increment theta by each iteration
@param iterations: the number of iterations to complete
'''
def linear_regression(train_X, train_y, X,
                      inital_theta=1, alpha=0.01, iterations=200) -> None:
    theta = gradient_descent(train_X, train_y, inital_theta, alpha, iterations)
    prediction = predict(X, theta)
    plt.plot(X, prediction)
    plt.scatter(train_X, train_y)
    plt.show()

