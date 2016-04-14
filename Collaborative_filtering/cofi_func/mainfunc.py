import code
import numpy as np
from loadMovieList import loadMovieList
#from costfunc import costfunc
from funcforfmincg import f,gradf
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

num_movies = Y.shape[0]
num_features = 10
num_users = Y.shape[1]

params = np.c_[X.reshape(1, num_movies*num_features), Theta.reshape(1, num_users*num_features)]
#print('size of params: \n', params.shape)
#(J, grad) = costfunc(params, Y, R, num_users, num_movies, num_features, 1.5)
#print('J: \n', J)
#print('grad: \n', grad)

#================== Use fmin_cg function =================

args = (Y, R, num_users, num_movies, num_features, 100)
final_Theta = optimize.fmin_cg(f, x0 = params, fprime = gradf, args = args)
print('final_Theta: \n', final_Theta)

pause = code.InteractiveConsole()
pause.raw_input(prompt = "Press Enter to Continue: ")
