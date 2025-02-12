import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Load iris data
iris = datasets.load_iris()
X = iris.data[:100, [0, 2]]  # sepal length and petal length
y = iris.target[:100]        # only first two classes
y = np.where(y == 0, -1, 1)  # convert to -1 and 1

# Add Gaussian noise to make data non-linearly separable
np.random.seed(1)  # for reproducibility
noise = np.random.normal(0, 0.5, X.shape)  # mean=0, std=0.5
X = X + noise

# Visualize perturbed data
plt.figure(figsize=(8,6))
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='red', marker='s', label='-1')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='x', label='1')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.title('Perturbed Iris Data')
plt.tight_layout()
plt.show()