import numpy as np

"""
This is the most intuitive way i know how to code a stochastic-based gradient descent algorithm 
for cost function optimization

"""

# Take a Very Basic Dataset with 3 samples in it
X = [[1, 2],    # sample 1
     [3, 4],    # sample 2
     [5, 6]]    # sample 3

y = [1, -1, 1]  # corresponding labels

w = [ 0.1, 0.1]
b = 0.1
# weight vector has the same number of elements as an ith feature within the feature vector
learning_rate = 0.001

def Stochastic_Method():

    # Take a Very Basic Dataset with 3 samples in it
    X = [[1, 2],    # sample 1
        [3, 4],    # sample 2
        [5, 6]]    # sample 3

    y = [1, -1, 1]  # corresponding labels

    w = [ 0.1, 0.1]
    b = 0.1
    # weight vector has the same number of elements as an ith feature within the feature vector
    learning_rate = 0.001

    for i in range(3): # 3 for each sample, this is different than an epoch(just one epoch in this instant)
             # Calculate the prediction for the ith sample being evaluated
             #Z is the net sum, which is what our dot product of feature vectors and weights and bias are
            z = w[0]*X[i][0] + w[1]*X[i][1] + b # Fix the weight index  so that it always evauates the w[0] against 0th(first) column in the feature vector

            #IMMEDIATELY AFTER, we update the weights after the ith sample has had its net sum calculated
            w[0] +=   learning_rate*(y[i] - z)*X[i][0] 
            w[1] +=   learning_rate*(y[i] - z)*X[i][1]
            b += b + learning_rate*(y[i] - z)

            print(f"After sample {i+1}, w = {w} and b = {b}")

Stochastic_Method()

"""
Discussion:

What characteristics of this makes the algorithm "Stochastic"?

The "one-at-a-time" approach is what makes it random/probabilistic in nature. The weights are updated 
after each random sample. Makes more frequent but less accurate updates. It is memory efficient - it doesn't need to
store gradients for the entire dataset. It can learn as new data comes in, it doesn't need the entire dataset.

Think of it like:

Stochastic GD: "Let me look at one peice of data, take a quick step, then move on to the next datapoint


"""