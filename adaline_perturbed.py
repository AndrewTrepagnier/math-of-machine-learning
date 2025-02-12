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


class AdalineGD(object):


    """ ADAptive LInear NEuron Classifier (ADALINE)

    we will take a basic perceptron algorithm and change the fit method so that the weights are updated by minimizing the cost function via gradient descent

    PARAMETERS
    ==========================

    eta : float 
        learning rate between 0.0 and 1.0
    
    n_iter : int
        passes over the training dataset
    random_state : int
        Random number generator seed for random weight initialization
        
    ATTRIBUTES 
    ============================
    
    w_ : 1d-array 
        weights after fitting 
    cost_ : list 
        sum-of-squares cost function value in each epoch


    """ 

    def __init__ (self, eta = 0.01, n_iter = 50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        For fititng the training data...


        PARAMETERS
        =========================

        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and n_features is the number of feature

        y : {array-like}, shape = [n_examples]
            target values (aka the true class label)

        RETURNS
        ==========================

        self : object 

        """
            # Example with 3 features (like height, weight, age)
            # X = np.array([
            #     [170, 70, 30],  # Person 1
            #     [160, 65, 25],  # Person 2
            #     [180, 75, 35]   # Person 3
            # ])

            # # So X.shape[1] = 3 (3 features)
            # # Therefore size = 1 + X.shape[1] = 4 weights needed
        
        

        randgen = np.random.RandomState(self.random_state) # generates our random seed

        self.w_ = randgen.normal(loc = 0.0, scale = 0.01, size = 1+ X.shape[1] ) # creates a weight matrix that is the same shape of X (feature vector)
        # For example: [w0, w1, w2, ...], where w0 is a weight for feature[0] and so on.
        # w_ Might output something like:
        # [ 0.00173  -0.00854   0.00583  -0.00277, ...

        self.cost_ = []

        """Training Loop"""
        for i in range(self.n_iter):
            # machine makes a prediction, look at net_input function and activation to better understand
            net_input = self.net_input(X)
            output = self.activation(net_input)

            errors = (y - output)
            # calculate error between true value and predicted value (how wrong was the machine?)

            self.w_[1:] += self.eta * X.T.dot(errors) 
            # Update feature weights:
            # - X.T.dot(errors) calculates how each feature contributed to the errors
            # - self.eta controls how much we adjust (learning rate)
            # - w_[1:] means all weights except bias

            self.w_[0] += self.eta * errors.sum()
            # Update bias weight separately:
            # - errors.sum() adds up all errors
            # - self.eta controls adjustment size
            # - w_[0] is the bias term


            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            # Track our progress:
            # - Square errors to make all errors positive
            # - Sum them up to get total error
            # - Divide by 2 (makes derivative simpler)
            # - Store in cost_ list to see if we're improving


        return self
    
    def net_input(self, X):
        """ Calculate the net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""

        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
fig, ax = plt.subplots(nrows = 1 , ncols = 2, figsize =(10,4))
ada1 = AdalineGD(n_iter=10, eta = 0.01).fit(X,y)

ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(SSE)")
ax[0].set_title('Adaline Learning rate = 0.01')

ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(X,y)

ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("log(SSE)")
ax[1].set_title('Adaline Learning rate = 0.0001')
    
plt.show()