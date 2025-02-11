import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris','iris.data')

print('URL', s)

#URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.DeprecationWarning

df = pd.read_csv(s, header=None, encoding='utf-8')

df.tail()

""" EXTRACT DATA FOR VISUAL"""

y = df.iloc[0:100, 4].values
y = np.where(y == 'IRIS-Setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

# plt.scatter(X[:50, 0], X[:50, 1], color='blue', marker='x', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='red', marker='o', label='versicolor')


# plt.show()