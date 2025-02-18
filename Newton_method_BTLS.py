import numpy as np


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
