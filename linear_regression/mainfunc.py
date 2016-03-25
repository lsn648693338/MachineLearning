import code
import numpy as np
import plotData

# ============================== Load and Plot data ==============================

print("Plotting Data ...\n")
data = np.genfromtxt("../data/ex1data1.txt", delimiter = ",")
X = data[:,0]
y = data[:,1]
plotData.plotData(X, y)

pause = code.InteractiveConsole()
pause.raw_input(prompt = "Press Enter to continue: ")

# ============================== Gradient descent ================================

print("Running Gradient Descent ...\n")

