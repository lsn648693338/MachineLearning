import numpy as np

def predfunc(final_Theta, num_users, num_movies, num_features, Ymean):
    X = final_Theta[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = final_Theta[num_movies*num_features:].reshape(num_users, num_features)

    predict = np.dot(X, Theta.T)
    my_predict = predict[:, 0].reshape(num_movies, 1) + Ymean
    return my_predict
