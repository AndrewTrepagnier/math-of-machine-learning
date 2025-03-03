import numpy as np
import matplotlib.pyplot as plt


X = ([0.0, 1.1] ,
    [1.0, 2.6] ,
    [2.0, 7.2],
    [3.0, 21.1])
X = np.array(X)   


def BatchGD(X,Y):

    learning_rate = 0.01

    

    w = np.random.uniform(0, 0.05, 2) # random values in specific range (0 to 0.05)
    b = np.random.uniform(0, 0.05, 1)

    delta_w = np.zeros(2)
    delta_b = 0
    

    for i in range(3): # 3 for each sample, this is different than an epoch(just one epoch in this instant)
        z = w[0]*X[i][0] + w[1]*X[i][1]

        
        # Below is the Cost Function Derivative, These will accumulate over each epoch by adding on top of the last 
        # Another word for this could be the "Gradient Accumulation" or "Error Gradients"
        delta_w[0] += (Y[i] - z)*X[i][0] 
        delta_w[1] += (Y[i] - z)*X[i][1]
        delta_b = (Y[i] - z)


    #After the third epoch, we can make the actual Weight Update
    w[0] += learning_rate*(delta_w[0])
    w[1] += learning_rate*(delta_w[1])
    b += learning_rate*(delta_b) 
    print(f"after sample {i} was complete, w = {w} and b = {b}")
    return



plt.scatter(X[:, 0], X[:, 1])  # X[:, 0] gets all x-coords, X[:, 1] gets all y-coords
plt.show()


"""
Discussion:

How can you tell these are adaline implementations?

1) There is linear activation in which Z = w[0]*X[i][0] + w[1]*X[i][1] + b 

        There is no step function like a perceptron (-1 or 1)

2) Continuous Error - it is based on the difference between the true class value and the net input (y[i] - z)
        Perceptron uses discrete predictions - step(z) will yield either 1 or -1. no inbetween


"""

