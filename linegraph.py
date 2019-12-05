import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("GNBResults2.csv")
df2 = pd.read_csv("MNBResults2.csv")
df3 = pd.read_csv("RFCResults2.csv")

fig = plt.figure()

ax = plt.axes()

plt.plot(df["SampleSize"], df["Accuracy"], color="k", label="G Naive Bayes")
plt.plot(df2["SampleSize"], df2["Accuracy"], color="b", label="M Naive Bayes")
plt.plot(df3["SampleSize"], df3["Accuracy"], color="g", label="Random Forest")

# Put a legend to the right of the current axis
ax.legend(loc="best")

plt.ylabel("Accuracy")
plt.xlabel("Sample Size")

plt.show()

