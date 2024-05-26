'''
@auther anott03
simple implementations of machine learning algorithms
'''

import numpy as np
from linear_regression import linear_regression
from logistic_regression import gradient_descent
import math

if __name__ == "__main__":
   #  train_X = np.array(range(0, 100, 2))
   #  train_y = np.array([x**2 for x in train_X])
   #  linear_regression(train_X, train_y, np.array(range(200)), iterations=1000)

   #  train_X = np.array(range(0, 100, 2))
   #  train_y = np.array([x**2 for x in range(0, 50, 2)] + [math.sqrt(x) for x in range(51, 100, 2)])
   #  linear_regression(train_X, train_y, np.array(range(200)), iterations=1000)

   train_X = np.array([0, 1, 2, 3, 4])
   train_y = np.array([1, 1, 1, 0, 0])
   print(gradient_descent(train_X, train_y, 0.01, 2, iterations=200))
