import pandas as pd
import matplotlib.pyplot as plt

input_path = r"C:\Users\geert\OneDrive - UCL\PhD\Probabilistic eval project St.Luc\ORL_001\ORL_001\test\results.csv"
data = pd.read_csv(input_path)
fig, ax = plt.subplots()
ax.axis("off")
ax.table(cellText=data.values, colLabels=data.columns, loc="center")
plt.show()