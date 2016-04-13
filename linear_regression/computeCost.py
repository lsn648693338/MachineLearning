import numpy as np

def computeCost(X, y, theta):
    m = len(y)
    J = 0
    J = (1/(2*m)) * np.dot(np.reshape((np.dot(X, theta) - y), (1, X.shape[0])), (np.dot(X, theta) - y))
    return J
