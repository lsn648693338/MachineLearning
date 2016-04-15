#import code
import numpy as np
from loadMovieList import loadMovieList
#from costfunc import costfunc
from funcforfmincg import f,gradf
from predfunc import predfunc
from normalizeRatings import normalizeRatings
#import plotData

from scipy import optimize

#print("Plotting Data ...\n")
Y = np.genfromtxt("../data/movies_Y.csv", delimiter = ',')
Y = Y[:, 0:-1]
#print('Y: \n', Y)
#plotData.plotData(Y)

X = np.genfromtxt("../data/movies_X.csv", delimiter = ',')
X = X[:, 0:-1]
#print('X: \n', X)

R = np.genfromtxt("../data/movies_R.csv", delimiter = ',')
R = R[:, 0:-1]
#print('R: \n', R)

Theta = np.genfromtxt("../data/movies_Theta.csv", delimiter = ',')
Theta = Theta[:, 0:-1]
#print('Theta: \n', Theta)

movielist = loadMovieList()
#print(movielist)

#print('size of params: \n', params.shape)
#(J, grad) = costfunc(params, Y, R, num_users, num_movies, num_features, 1.5)
#print('J: \n', J)
#print('grad: \n', grad)

#================== Entering ratings for a new user ================

my_rating = np.zeros([Y.shape[0], 1])

my_rating[0, 0] = 4
my_rating[97, 0] = 2
my_rating[6, 0] = 3
my_rating[11, 0] = 5
my_rating[53, 0] = 4
my_rating[63, 0] = 5
my_rating[65, 0] = 3
my_rating[68, 0] = 5
my_rating[182, 0] = 4
my_rating[225, 0] = 5
my_rating[354, 0] = 5

for i in range(Y.shape[0]):
    if my_rating[i, 0] != 0:
        print('Rated %d for %s' % (my_rating[i, 0], movielist[i+1]))

#================== Add my_rating to training set ==================

Y = np.c_[my_rating, Y]
my_R = (my_rating != 0)
my_R.dtype = 'int8'
R = np.c_[my_R, R]

num_movies = Y.shape[0]
num_features = 10
num_users = Y.shape[1]

X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

params = np.c_[X.reshape(1, num_movies*num_features), Theta.reshape(1, num_users*num_features)]

Ynorm, Ymean = normalizeRatings(Y, R)
print('Ynorm: \n', Ynorm[0,0], Ynorm[1,1])
#================== Use fmin_cg function ===========================

args = (Ynorm, R, num_users, num_movies, num_features, 10)
final_Theta = optimize.fmin_cg(f, x0 = params, fprime = gradf, args = args)
#print('final_Theta: \n', final_Theta)

my_predict = predfunc(final_Theta, num_users, num_movies, num_features, Ymean)
print(my_predict)

#================== Sort the prediction ============================

row_predict = my_predict.reshape(1, num_movies)[0]
sorted_predict_indices = row_predict.argsort()[::-1]
sorted_predict = np.sort(row_predict)[::-1]

for i in range(10):
    j = sorted_predict_indices[i]
    print('Predicting rating %.1f for movie %s\n' % (row_predict[j], movielist[j+1]))

#print('row_predict: \n', row_predict)
#print('predict: \n', sorted_predict)
#print('predict indices: \n', sorted_predict_indices)
#print('predict: \n', predict)
#print('max of predict: \n', predict.max())
#print('min of predict: \n', predict.min())

#plotData.plotData(predict)

#pause = code.InteractiveConsole()
#pause.raw_input(prompt = "Press Enter to Continue: ")
