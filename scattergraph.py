import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("GNBResults.csv")
df2 = pd.read_csv("MNBResults.csv")
df3 = pd.read_csv("RFCResults.csv")

fig = plt.figure()

ax = plt.axes()

plt.scatter(df["Accuracy"], (df["ChunkSize"]*df2["Chunks"]), color="k", label="G Naive Bayes")
plt.scatter(df2["Accuracy"], (df2["ChunkSize"]*df2["Chunks"]), color="b", label="M Naive Bayes")
plt.scatter(df3["Accuracy"], (df3["ChunkSize"]*df3["Chunks"]), color="g", label="Random Forest")

# Plot base random divider
plt.plot([0.05, 0.05], plt.gca().get_ylim(), color='r', linestyle='-', linewidth=2)

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.79, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel("Accuracy")
plt.ylabel("Chunk Magnitude")

plt.show()

