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

Added random guassian noise such that the data is no longer linearly seperable  

Added L2 regularization to compare performance on noisy data

"""

# Load iris data
iris = datasets.load_iris()
X = iris.data[:100, [0, 2]]  # sepal length and petal length
y = iris.target[:100]        # only first two classes
y = np.where(y == 0, -1, 1)  # convert to -1 and 1

# Create a copy of the original data for comparison
X_original = np.copy(X)

# Add stronger noise with directional bias
np.random.seed(1)  # for reproducibility
noise = np.random.normal(0, 1.5, X.shape)  # Increased std to 1.5
# Add directional bias to some points to make them cross the boundary
for i in range(len(X)):
    if np.random.rand() < 0.2:  # 20% of points get extra shift
        noise[i] *= 2.5  # Make noise stronger for these points
        
X = X + noise

# Standardization
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# Also standardize original data for comparison
X_original_std = np.copy(X_original)
X_original_std[:,0] = (X_original[:,0] - X_original[:,0].mean()) / X_original[:,0].std()
X_original_std[:,1] = (X_original[:,1] - X_original[:,1].mean()) / X_original[:,1].std()

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
    
    lambda_param : float
        regularization strength (0 = no regularization)
        
    random_state : int
        Random number generator seed for random weight initialization
        
    ATTRIBUTES 
    ============================
    
    w_ : 1d-array 
        weights after fitting 
    cost_ : list 
        sum-of-squares cost function value in each epoch


    """ 

    def __init__ (self, eta = 0.01, n_iter = 50, lambda_param=0.0, random_state=1):
        self.n_iter = n_iter
        self.eta = eta  #learning rate
        self.lambda_param = lambda_param  # regularization parameter
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
        randgen = np.random.RandomState(self.random_state) # generates our random seed

        self.w_ = randgen.normal(loc = 0.0, scale = 0.01, size = 1+ X.shape[1] ) # creates a weight matrix that is the same shape of X (feature vector)

        self.cost_ = []

        """Training Loop"""
        for i in range(self.n_iter):
            # machine makes a prediction, look at net_input function and activation to better understand
            net_input = self.net_input(X)
            output = self.activation(net_input)

            errors = (y - output)
            # calculate error between true value and predicted value (how wrong was the machine?)

            # Update weights with regularization for all except bias
            self.w_[1:] += self.eta * (X.T.dot(errors) - self.lambda_param * self.w_[1:])
            
            # Bias is not regularized
            self.w_[0] += self.eta * errors.sum()
            
            # Cost function with L2 regularization
            cost = -np.sum(y*np.log(output) + (1-y)*np.log(1-output)) + \
                   (self.lambda_param/2) * np.sum(self.w_[1:]**2)
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
    
# Train model on original data
original_model = logisticregGD(n_iter=30, eta=0.01)
original_model.fit(X_original_std, y)

# Train model on noisy data without regularization
noisy_model = logisticregGD(n_iter=30, eta=0.01)
noisy_model.fit(X_std, y)

# Train model on noisy data with L2 regularization
reg_model = logisticregGD(n_iter=30, eta=0.01, lambda_param=0.1)
reg_model.fit(X_std, y)

# Create plots - now with 6 subplots to compare all models
plt.figure(figsize=(15, 12))

# Plot 1: Original Data Decision Regions
plt.subplot(3, 2, 1)
plot_decision_regions(X_original_std, y, classifier=original_model)
plt.title('Original Data - Decision Boundary')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

# Plot 2: Original Data Cost Function
plt.subplot(3, 2, 2)
plt.plot(range(1, len(original_model.cost_) + 1), original_model.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Original Data - Cost Function')

# Plot 3: Noisy Data Decision Regions
plt.subplot(3, 2, 3)
plot_decision_regions(X_std, y, classifier=noisy_model)
plt.title('Noisy Data - Decision Boundary')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

# Plot 4: Noisy Data Cost Function
plt.subplot(3, 2, 4)
plt.plot(range(1, len(noisy_model.cost_) + 1), noisy_model.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Noisy Data - Cost Function')

# Plot 5: Regularized Model Decision Regions
plt.subplot(3, 2, 5)
plot_decision_regions(X_std, y, classifier=reg_model)
plt.title('Noisy Data with L2 Regularization - Decision Boundary')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

# Plot 6: Regularized Model Cost Function
plt.subplot(3, 2, 6)
plt.plot(range(1, len(reg_model.cost_) + 1), reg_model.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Noisy Data with L2 Regularization - Cost Function')

plt.tight_layout()
plt.show()


