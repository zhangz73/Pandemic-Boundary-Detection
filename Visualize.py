import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../Results/BoundaryResultsGeneral.csv")
#df = df[(df["T"] == 100) & (df["Population"] == 1000000) & (df["p"] == 0.1) & (df["NonCooperativeProb"] == 0)]
df = df[(df["p"] == 0.1) & (df["Population"] == 1000000) & (df["NonCooperativeProb"] == 0)]

type = "Cell"
df = df.sort_values(f"TotalTest{type}")
print(df)

denom_DFS = list(np.array(df[df.Algorithm == "DFS"][f"ExternalBoundary{type}"]) + np.array(df[df.Algorithm == "DFS"][f"Internal{type}"]))
test_DFS = list(np.array(df[df.Algorithm == "DFS"][f"TotalTest{type}"]))

lists = sorted(zip(*[denom_DFS, test_DFS]))
x, y = list(zip(*lists))

plt.plot(x, y)
plt.show()
