import code
import numpy as np
import plotData
import computeCost
import gradientDescent

# ============================== Load and Plot data ==============================

print("Plotting Data ...\n")
data = np.genfromtxt("../data/ex1data1.txt", delimiter = ",")
X = data[:, 0]
y = data[:, 1]
plotData.plotData(X, y)

pause = code.InteractiveConsole()
pause.raw_input(prompt = "Press Enter to continue: ")

# ============================== Gradient descent ================================

print("Running Gradient Descent ...\n")
m = len(y)
X = np.c_[np.ones((m, 1)), data[:, 0]]
X = np.reshape(X, (m, 2))
y = np.reshape(y, (m, 1))
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

temp = computeCost.computeCost(X, y, theta)
print("The first J: ", temp)

[theta, J] = gradientDescent.gradientDescent(X, y, theta, alpha, iterations)
print("Theta found by gradient descent: ")
print("%f %f \n" % (theta[0], theta[1]))
print("The sequence of J: \n")
print(J)
