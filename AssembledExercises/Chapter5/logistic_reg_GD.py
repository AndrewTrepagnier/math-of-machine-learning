import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from matplotlib.colors import ListedColormap


"""
Modified the adaline gradient descent algorithm with a standardized dataset, Xsd, and applied
a logistic regression gradient descent method

This will be the same as adaline except it will apply the standard logisitic sigmoid function as the activation 
function rather than the identity function

"""

# Load iris data
iris = datasets.load_iris()
X = iris.data[:100, [0, 2]]  # sepal length and petal length
y = iris.target[:100]        # only first two classes
y = np.where(y == 0, -1, 1)  # convert to -1 and 1

# Standardization
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x')  
    colors = ('red', 'blue')  
    cmap = ListedColormap(colors)

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                   y=X[y == cl, 1],
                   alpha=0.8, 
                   c=colors[idx],
                   marker=markers[idx],
                   label=cl,
                   edgecolor='black')

#Original Adaline with altered activation function
class logisticregGD(object):

    """
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
        self.n_iter = n_iter
        self.eta = eta  #learning rate
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
            

            self.w_[0] += self.eta * errors.sum()
            

            #COST FUCNTION CHANGE - 
            cost = -np.sum(y*np.log(output) + (1-y)*np.log(1-output))
            self.cost_.append(cost)
           


        return self
    
    def net_input(self, X):
        """ Calculate the net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute Nonlinear activation"""
        #Standard logistic sigmoid function
        return 1/(1+ np.exp(-X))
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, -1)
    
    
nonlinear_gd = logisticregGD(n_iter=15, eta=0.01)
nonlinear_gd.fit(X_std,y)

# Create plots
plt.figure(figsize=(10,4))

# Plot 1: Decision Regions
plt.subplot(1,2,1)
plot_decision_regions(X_std, y, classifier=nonlinear_gd)
plt.title('Logistic Regression')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

# Plot 2: Cost function
plt.subplot(1,2,2)
plt.plot(range(1, len(nonlinear_gd.cost_) + 1), nonlinear_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum squared error')
plt.tight_layout()
plt.show()


