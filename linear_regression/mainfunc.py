import code
import pandas as pd
import matplotlib.pyplot as plt

print("Plotting Data ...\n")
data = pd.read_csv('../data/ex1data1.txt', names = ["area", "price"])
print(data.head(5))
plt.scatter(data["area"], data["price"], color = "red")
plt.show()
pause = code.InteractiveConsole()
pause.raw_input(prompt = "Press Enter to continue: ")

