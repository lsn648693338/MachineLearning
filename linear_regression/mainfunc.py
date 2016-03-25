import code
import pandas as pd
import plotData

print("Plotting Data ...\n")
data = pd.read_csv('../data/ex1data1.txt', names = ["area", "price"])
print(data.head(5))
X = data["area"]
y = data["price"]
plotData.plotData(X, y)
pause = code.InteractiveConsole()
pause.raw_input(prompt = "Press Enter to continue: ")

