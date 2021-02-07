'''
@auther anott03
a simple logistic regression algorithm (that will get less simple over time)
'''

import numpy as np
import matplotlib.pyplot as plt
import math
from math import log

def sigmoid(z):
    return 1 / (1 + math.e**(-1*z))

def cost(h, _y):
    return (-1*_y * log(h)) - ((1-_y)*log(1 - h))

def compute_cost(X, y, theta):
    m = len(y)
    J = 0
    for i in range(m):
        h = sigmoid(X[i])
        J += cost(h, y[i])

    J /= m
    return J

def gradient_descent(X, y, alpha, inital_theta, iterations=200):
    theta = inital_theta
    m = len(y)
    # TBD
