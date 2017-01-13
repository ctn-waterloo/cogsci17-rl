import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

datafilename = 'data.txt'
if len(sys.argv) == 2:
    datafilename = sys.argv[1]

print(datafilename)
datafile = open(datafilename, "r").readlines()
all_data = np.zeros((len(datafile),4))

for i, line in enumerate(datafile):
        total = line.split(" ")
        all_data[i,:] = total

df = pd.DataFrame(data=all_data)
sns.barplot(data=df)
plt.show()
