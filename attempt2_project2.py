import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap




# Load in the wine dataset
wine = load_wine()
X = wine.data[:178]
print(np.shape(X)) #Shows us that there are 178 datapoints with 13 different features
y = wine.target[:178]  # Use the target labels instead of the data
y = np.where(y == 0, -1, 1)
print(np.shape(y))

# Standardization
X_std = np.copy(X)
X_std5 = np.copy(X)
X_std10 = np.copy(X)
X_std20 = np.copy(X)
for i in range(12):
    X_std5[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
    X_std10[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
    X_std20[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()

# Create a mask 
mask5 = np.random.random(X_std5.shape) < 0.05 # this is a boolean array with 5% true and the rest false
mask10 = np.random.random(X_std10.shape) < 0.1
mask20 = np.random.random(X_std20.shape) < 0.2
X_std5[mask5] = np.nan #When we use this mask with X_std[mask] = np.nan, it replaces all values where the mask is True with np.nan, effectively creating missing values in about 5% of the positions in your dataset.
X_std10[mask10] = np.nan
X_std20[mask20] = np.nan #Since the random numbers are uniformly distributed between 0 and 1, approximately 20% of them will be less than 0.2

# Convert y arrays to float type before setting NaN values
y5 = np.copy(y).astype(float)
y10 = np.copy(y).astype(float)
y20 = np.copy(y).astype(float)

# Check if any value in a row is NaN and update corresponding y value
for i in range(len(y)):
    if np.any(np.isnan(X_std5[i])):
        y5[i] = np.nan
    
    if np.any(np.isnan(X_std10[i])):
        y10[i] = np.nan
    
    if np.any(np.isnan(X_std20[i])):
        y20[i] = np.nan

clf0 = LogisticRegression(random_state=0).fit(X_std, y)
clf5 = LogisticRegression(random_state=0).fit(X_std5, y5)
clf10 = LogisticRegression(random_state=0).fit(X_std10, y10)


clf0.predict(X_std[:2, :])
clf5.predict(X_std5[:2, :])
clf10.predict(X_std10[:2, :])


print(f"Accuracy of SKlearn Logistic Regression with a L2 penalty and Zero missing data: {clf0.score(X_std, y)}")
print(f"Accuracy of SKlearn Logistic Regression with a L2 penalty and 5% missing data: {clf5.score(X_std5, y)}")
print(f"Accuracy of SKlearn Logistic Regression with a L2 penalty and 10% missing data: {clf10.score(X_std10, y)}")
