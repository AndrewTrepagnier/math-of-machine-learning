from scipy import optimize
import numpy as np

#===============================================

# what happens with X = [1, 2]
X = [1, 2]

# Manual calculation to see each term:
x1, x2 = X
term1 = 100 * (x2 - x1**2)**2  # 100(x₂ - x₁²)²
term2 = (1 - x1)**2            # (1 - x₁)²
total = term1 + term2

print(f"Term 1: 100(x₂ - x₁²)² = {term1}")
print(f"Term 2: (1 - x₁)² = {term2}")
print(f"Total: {total}")
print(f"Scipy rosen: {optimize.rosen(X)}")

"""
What rosen does with two entry elements is create a banana shaped valley in 2D.

The global minimum is at [1,1]

While finding the valley is easy, converging to the minimum is hard

Its a great test for optimization algorithms because the minimum requires both parameters to be optimized together
"""

#================================================

# X = [1, 2, 3, 4, 5]  # example list

# # Positive slicing
# X[1:]   # [2, 3, 4, 5]     # from index 1 to end
# X[:1]   # [1]              # from start to index 1 (exclusive)
# X[1:3]  # [2, 3]          # from index 1 to 3 (exclusive)

# # Negative slicing
# X[:-1]  # [1, 2, 3, 4]     # from start to last element (exclusive)
# X[-1:]  # [5]              # just the last element
# X[-2:]  # [4, 5]          # last two elements

# # For the Rosenbrock function:
# x = [1, 2, 3]
# x[:-1]  # [1, 2]          # all elements except the last
# x[1:]   # [2, 3]          # all elements except the first

#===============================================
# GRADIENT DESCENT ALGORITHM WITH BACKTRACKING LINE SEARCH  
#===============================================

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

def optimize(x1, x2):
    X0 = np.array([x1, x2])  # Convert to numpy array
    gamma0 = 0.3
    epochs = 10
    
    print(f"Starting at X0 = {X0}")
    
    for n in range(epochs):
        print("=========================================================")
        print(f"\nEpoch {n+1}:")
        while True:
            print(f"  Current X = {X0}, γ = {gamma0}")  # Debug print
            if f(X0 - gamma0*f_grad(X0)) <= f(X0) - (gamma0/2)*(np.linalg.norm(f_grad(X0)))**2:
                X0 = X0 - gamma0*f_grad(X0)
                print(f"  → Step accepted, new X = {X0}")  # Debug print
                break
            else: 
                gamma0 = gamma0/2
                print(f"  → Step rejected, reducing γ to {gamma0}")  # Debug print
    
    return X0

# Run the optimization
result = optimize(5,4)
print(f"Final result: {result}")
         

