import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


# s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris','iris.data')

# print('URL', s)

# #URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.DeprecationWarning

# df = pd.read_csv(s, header=None, encoding='utf-8')

# df.tail()

# """ EXTRACT DATA FOR VISUAL"""

# y = df.iloc[0:100, 4].values
# y = np.where(y == 'IRIS-Setosa', -1, 1)

# X = df.iloc[0:100, [0,2]].values

iris = datasets.load_iris()
X = iris.data[:100, [0, 2]]  # sepal length and petal length
y = iris.target[:100]        # only first two classes
y = np.where(y == 0, -1, 1)  # convert to -1 and 1


print(X)
print(y)

# Visualize raw data
plt.figure(figsize=(8,6))
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='red', marker='s', label='-1')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='x', label='1')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.title('Raw Iris Data')
plt.tight_layout()
plt.show()