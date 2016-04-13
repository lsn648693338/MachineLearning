import numpy as np
import computeCost

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations, 1))
    for iter in range(iterations):
        theta = theta - alpha * (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
        J_history[iter] = computeCost.computeCost(X, y, theta)
    return theta, J_history
