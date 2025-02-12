import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

class Ppn_to_ada():
    def __init__(self, xdim, epoch=10, learning_rate=0.15): # used a larger learning rate since i have standardized data
        self.epoch = epoch
        self.learning_rate = learning_rate
        # self.weights = np.zeros(xdim + 1) archived*******
         # +1 for bias weight
         ############################################ makes weight matrix of non-zero, small values
        rng = np.random.RandomState(1)  # seed = 1
        self.weights = rng.uniform(0, 0.1, X.shape[1] + 1)     
        
        ###########################################
    # def activate(self, x):
    #     net_input = np.dot(x, self.weights[1:]) + self.weights[0]
    #     return 1 if (net_input > 0) else 0

    def activation(self, X):
        return X # passes the whatever the input is as itself, in this instant, it will be net input(XdotW)
    
    # def fit(self, Xtrain, ytrain):
    #     self.errors_ = []
    #     for k in range(self.epoch):
    #         errors = 0
    #         for x, y in zip(Xtrain, ytrain):
    #             yhat = self.activate(x)
    #             update = self.learning_rate * (y - yhat)
    #             self.weights[1:] += update * x
    #             self.weights[0] += update
    #             errors += int(update != 0.0)
    #         self.errors_.append(errors)
    #     return self
    #+++++++++++++++++++++++++++++++++++++++++++++++++++ New fit to convert to adaline
    def fit(self, Xtrain, ytrain):
        self.errors_ = []
        self.cost_ = []
        N = len(Xtrain)
        
        for k in range(self.epoch):
            output = np.dot(Xtrain, self.weights[1:]) + self.weights[0]  # Linear output
            errors = ytrain - output
            
            # Update weights
            self.weights[1:] += self.learning_rate * (1/N) * Xtrain.T.dot(errors)
            self.weights[0] += self.learning_rate * (1/N) * errors.sum()
            
            cost = (1/(2*N)) * (errors**2).sum()
            self.cost_.append(cost)
        return self
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def predict(self, Xtest):
        net_input = np.dot(Xtest, self.weights[1:]) + self.weights[0]
        return np.where(self.activation(net_input) >= 0.0, 1, -1)
    
    def score(self, Xtest, ytest):
        count = 0
        for x, y in zip(Xtest, ytest):
            yhat = self.activate(x)
            if yhat == y:
                count += 1
        return count / len(ytest)

# Load iris data
data_read = datasets.load_iris()
X = data_read.data[:100, [0, 2]]  # sepal length and petal length
y = data_read.target[:100]        # only first two classes
y = np.where(y == 0, -1, 1)      

# After loading iris data
# Standardize the features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
X = X_std  # Use standardized features


plt.figure(figsize=(10,6))

# Plot the data points
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='o', label='setosa')
plt.scatter(X[y==1, 0], X[y==1, 1], color='orange', marker='x', label='versicolor')

# Train Adaline and plot final decision boundary
clf = Ppn_to_ada(X.shape[1], epoch=50)  # or whatever number of epochs you want
clf.fit(X, y)

# Create decision boundary line
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x = np.array([x_min, x_max])
w = clf.weights
if w[2] != 0:  # Avoid division by zero
    y_boundary = (-w[0] - w[1]*x) / w[2]  # Decision boundary equation
    plt.plot(x, y_boundary, color='red', label='Final Decision Boundary')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend()
plt.grid(True)
plt.title('Adaline Decision Boundary')
plt.show()

# Plot convergence
plt.figure(figsize=(8,6))
plt.plot(range(1, len(clf.errors_) + 1), clf.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.title('Perceptron Learning Convergence')
plt.grid(True)
plt.show()