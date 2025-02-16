import numpy as np
import matplotlib.pyplot as plt



# gamma = 0.01
# X= []

# def my_gradfunction(X): #for quadratic
#     return 2*X

# for i in range(1000):
#     x_ = i + gamma*my_gradfunction(i)
#     x_next =+ x_
# print(x_next)

# Define the function and its gradient


def f(X_): # i only designed this for 2D optimization problems so input vector is only two elements
    result = 100 * (X_[1] - X_[0]**2)**2 + (1 - X_[0])**2
    print(f"f(X) = {result}")  # Debug print
    return result

def f_grad(X_): 
    df_x1 = -400*X_[0]*(X_[1] - X_[0]**2) -2 * (1- X_[0])
    df_x2 =  200 * (X_[1] - X_[0]**2) 
    grad = np.array([df_x1, df_x2])  # Convert to numpy array
    print(f"gradient = {grad}")  # Debug print
    return grad


def optimize(x1, x2, g):
    X_ = np.array([x1,x2])
    gamma = g
    epochs = 10
    
    print(f"Starting at X = {X_}")
    print(f"Initial function value = {f(X_)}")
    
    for n in range(epochs):
        grad = f_grad(X_)
        X_new = X_ - gamma*(grad)
        
        # Check for explosion
        if np.any(np.abs(X_new) > 1e10) or np.any(np.isnan(X_new)):
            print("\n⚠️ EXPLOSION DETECTED!")
            print(f"  Iteration {n}")
            print(f"  Previous X = {X_}")
            print(f"  Gradient = {grad}")
            print(f"  Step = {gamma*grad}")
            print(f"  Exploded to X = {X_new}")
            return X_  # Return last valid position
            
        print(f"\nEpoch {n}:")
        print(f"  Gradient = {grad}")
        print(f"  Step size = {gamma}")
        print(f"  New X = {X_new}")
        print(f"  New function value = {f(X_new)}")
        
        X_ = X_new
        
    return X_

print(f"Final Result: {optimize(5,4, 0.01)}")

print("++++++++++++++++++++++++++++Smaller Step++++++++++++++++++")

print(f"Final Result: {optimize(5,4, 0.0001)}")