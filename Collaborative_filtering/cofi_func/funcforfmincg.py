import numpy as np

def f(x, *args):
    J = 0
    Y, R, num_users, num_movies, num_features, lambda1 = args

    X = x[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = x[num_movies*num_features:].reshape(num_users, num_features)

    temp = (np.dot(X, Theta.T) - Y) * R
    J = 1/2 * (temp**2).sum() + lambda1/2 * (Theta**2).sum() + lambda1/2 * (X**2).sum()

    return J

def gradf(x, *args):
    Y, R, num_users, num_movies, num_features, lambda1 = args

    X = x[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = x[num_movies*num_features:].reshape(num_users, num_features)

    temp = (np.dot(X, Theta.T) - Y) * R
    X_grad = np.dot(temp, Theta) + np.dot(lambda1, X)
    Theta_grad = np.dot(temp.T, X) + np.dot(lambda1, Theta)

    grad = np.c_[X_grad.reshape(1, X_grad.size), Theta_grad.reshape(1, Theta_grad.size)]

    return grad[0]
