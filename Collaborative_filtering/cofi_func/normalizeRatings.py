import numpy as np

def normalizeRatings(Y, R):
    m, n = Y.shape[0], Y.shape[1]
    Ymean = np.zeros([m, 1])
    Ynorm = np.zeros(Y.shape)

    for i in range(m):
        indices = np.nonzero(R[i, :] == 1)
        Ymean[i] = Y[i, indices].mean()
        Ynorm[i, indices] = Y[i, indices] - Ymean[i]
    return Ynorm, Ymean
