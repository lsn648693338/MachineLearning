import numpy as np

def costfunc(params, Y, R, num_users, num_movies, num_features, lambda1):
    J = 0
    X = params[0, 0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[0, num_movies*num_features:].reshape(num_users, num_features)
    print(type(params))
    print('size of X: \n', X.shape, type(X))
    print('size of Theta: \n', Theta.shape)

    temp = (np.dot(X, Theta.T) - Y) * R
    J = 1/2 * (temp**2).sum() + lambda1/2 * (Theta**2).sum() + lambda1/2 * (X**2).sum()
    X_grad = np.dot(temp, Theta) + np.dot(lambda1, X)
    Theta_grad = np.dot(temp.T, X) + np.dot(lambda1, Theta)

    grad = np.r_[X_grad.reshape(X_grad.size, 1), Theta_grad.reshape(Theta_grad.size, 1)]
    print('size of X_grad: \n', X_grad.shape)
    print('size of Theta_grad: \n', Theta_grad.shape)
    print('X: \n', X)
    print('Theta: \n', Theta)
    return J, grad
