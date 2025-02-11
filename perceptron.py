import numpy as np
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

