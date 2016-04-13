import code
import numpy as np
from loadMovieList import loadMovieList
import plotData

print("Plotting Data ...\n")
Y = np.genfromtxt("../data/movies_Y.csv", delimiter = ',')
Y = Y[:, 0:-1]
print(Y)
plotData.plotData(Y)

X = np.genfromtxt("../data/movies_X.csv", delimiter = ',')
X = X[:, 0:-1]
print(X)

R = np.genfromtxt("../data/movies_R.csv", delimiter = ',')
R = R[:, 0:-1]
print(R)

Theta = np.genfromtxt("../data/movies_Theta.csv", delimiter = ',')
Theta = Theta[:, 0:-1]
print(Theta)

movielist = loadMovieList()
print(movielist)

pause = code.InteractiveConsole()
pause.raw_input(prompt = "Press Enter to Continue: ")
