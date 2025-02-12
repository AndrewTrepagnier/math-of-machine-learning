import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
np.set_printoptions(suppress=True)



"""
Perceptron from course
"""
class Perceptron_course():
    def __init__(self, xdim, epoch=10, learning_rate=0.01):
        """
        Initialize perceptron with:
        xdim: number of input features
        epoch: number of training iterations
        learning_rate: how fast the model learns
        """
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weights = np.zeros(xdim + 1)  # +1 for bias weight
    
    def activate(self, x):
        """
        Activation function - step function
        Returns 1 if weighted sum > 0, else 0
        """
        net_input = np.dot(x, self.weights[1:]) + self.weights[0]
        return 1 if (net_input > 0) else 0
    
    def fit(self, Xtrain, ytrain):
        """
        Train the perceptron:
        Xtrain: training features
        ytrain: training labels
        """
        for k in range(self.epoch):
            for x, y in zip(Xtrain, ytrain):
                yhat = self.activate(x)  # predict
                # Update weights if prediction is wrong
                self.weights[1:] += self.learning_rate * (y - yhat) * x
                self.weights[0] += self.learning_rate * (y - yhat)
    
    def predict(self, Xtest):
        """
        Make predictions on test data
        Returns list of predictions
        """
        yhat = []
        [yhat.append(self.activate(x)) for x in Xtest]
        return yhat
    
    def score(self, Xtest, ytest):
        """
        Calculate accuracy of predictions
        Returns percentage correct
        """
        count = 0
        for x, y in zip(Xtest, ytest):

            yhat = self.activate(x)
            if yhat == y:
                count += 1
        return count / len(ytest)


#====================================


    def fit_and_fig(self, Xtrain, ytrain):

        wgts_all = []
        for k in range(self.epoch):
            for x,y in zip(Xtrain, ytrain):
                yhat = self.activate(x)
                self.weights[1:] += self.learning_rate*(y-yhat)*x
                self.weights[0] += self.learning_rate*(y-yhat)
                if k ==0: wgts_all.append(list(self.weights))
        return np.array(wgts_all)
    
#+++++++++++++++++++++++++++++++++++++++++
print(dir(datasets))

data_read = datasets.load_iris()
X = data_read.data
y = data_read.target

targets = data_read.target_names
features = data_read.feature_names

N,d = X.shape
nclass = len(set(y))

#_______________________Take 2 classes in 2D_____________
X2 = X[y<=1]
y2 = y[y<=1]
X2 = X2[:, [0,2]]

#________________________Train and Test___________________


Xtrain, Xtest, ytrain, ytest = train_test_split(X2, y2, random_state = None, train_size = 0.7e0)

clf = Perceptron_course(X2.shape[1], epoch=2)
clf.fit(Xtrain,ytrain)
wgts_all = clf.fit_and_fig(Xtrain, ytrain)
accuracy = clf.score(Xtest,ytest)
print('accuracy = ', accuracy)
yhat = clf.predict(Xtest)

