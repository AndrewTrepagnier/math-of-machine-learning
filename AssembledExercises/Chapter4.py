import numpy as np
import matplotlib.pyplot as plt

#+++++++++++++++++++++++++++++++++++STARTING WITH DEFINING FUNCTIONS+++++++++++++++++++++++++++++++++++++++++
"""
Rosenbrock Function is infamous for testing optimization algorithms. I implement the Rosen below manually
"""
def f(X_): # i only designed this for 2D optimization problems so input vector is only two elements
    result = 100 * (X_[1] - X_[0]**2)**2 + (1 - X_[0])**2
   # print(f"f(X) = {result}") 
    return result

# First Gradient of the Rosen Function
def f_grad(X_):  
    df_x1 = -400*X_[0]*(X_[1] - X_[0]**2) -2 * (1- X_[0])
    df_x2 = 200 * (X_[1] - X_[0]**2) 
    grad = np.array([df_x1, df_x2])
    # print(f"gradient = {grad}")
    return grad

# The hessian is a matrix of second order gradients of the rosen wrt different values. This is valuable for defining local curvature of a function at a point
def _hess_(X_):
    df_2_x1 = 800*X_[0]**2 - 400*(X_[1] - X_[0]**2) + 2
    df_2_x2 = 200
    df_2_x1x2 = -400*X_[0]
    hessian = np.array([[df_2_x1, df_2_x1x2],
                       [df_2_x1x2, df_2_x2]])
    #to return inverse hessian just use this np.linalg.inv(hessian)
    return hessian

def _inverse_hess_(X_):
    return np.linalg.inv(_hess_(X_))

def approximated_hess_(X_):
    df_2_x1 = 800*X_[0]**2 - 400*(X_[1] - X_[0]**2) + 2
    df_2_x2 = 200
    diag = np.array([ df_2_x1, 0],
   [0, df_2_x2])
    return approximated_hess_

def _gradient_descent_(X_, gamma_parameter):
    for _ in range(epochs):
        xn_next =  X_ - gamma_parameter*f_grad(X_)
        X_ = xn_next
        stored_Xs.append(X_)
    return X_, stored_Xs # Returns the final position after all epochs and the array of past positions  


def GD_with_BTLS(X_, gamma_parameter): # calling the input Xn just for easier readability, it will still be X_ that is input into the function
    stored_Xs=[]
    for _ in range(epochs):
        while True:
            if  f(X_ - gamma_parameter*f_grad(X_)) <= f(X_) - (gamma_parameter/2)*(np.linalg.norm(f_grad(X_))**2): #Euclidean norm is the same as grad @ grad 
                X_ =  X_ - gamma_parameter*f_grad(X_) #as you can see this is gradient descent method
                stored_Xs.append(X_)
                break
            else:
                gamma_parameter = gamma_parameter /2 
    return stored_Xs

def Newton_with_BTLS(X_, gammma_parameter): # everything is the same as gradient descent except we use inverse hessian times gradient
    stored_Xs=[]
    for _ in range(epochs):
        while True:
            if  f(X_ - gamma_parameter*f_grad(X_)) <= f(X_) - (gamma_parameter/2)*(_inverse_hess_(X_) @ (np.linalg.norm(f_grad(X_))**2)): #notice inverse hessian
                X_ =  X_ - gamma_parameter*f_grad(X_) 
                stored_Xs.append(X_)
                break
            else:
                gamma_parameter = gamma_parameter /2 
    return stored_Xs



def estimate_gamma(X_):
    """
    Estimates initial step size based on gradient magnitude.
    Smaller steps for steeper gradients, larger steps for gentler slopes.
    
    Args:
        X_: Current point (numpy array)
    Returns:
        float: Estimated step size gamma
    """
    grad = f_grad(X_)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm > 0:
        gamma = 1.0 / grad_norm
    else:
        gamma = 1.0  # Default if gradient is zero
        
    return gamma



#+++++++++++++++++++++++++++++++++++INITIALIZE+++++++++++++++++++++++++++++++++++++++++
X_ = [0.5, -0.5] # Starting point we will optimize FROM

#Carefully select one of these as your optimize argument if you want effective guess of gamma or not
gam = 0.01 # appropriate starting step size
gam_effect = estimate_gamma(X_)

epochs = 20 # keep this consistent across each problem tested
stored_Xs = [X_] # Placing starting point in the storage array








#+++++++++++++++++++++++++++++++++++SETUP PLOTTING SCHEME+++++++++++++++++++++++++++++++++++++++++






#+++++++++++++++++++++++++++++++++++CALL ON OPTIMIZATION FUNCTIONS+++++++++++++++++++++++++++++++++++++++++




#+++++++++++++++++++++++++++++++++++PROBLEM 1 
