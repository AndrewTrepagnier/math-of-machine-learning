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

    history = [X_]  # Initialize history list with starting point
    
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
            return X_, history  # Return history too
            
        print(f"\nEpoch {n}:")
        print(f"  Gradient = {grad}")
        print(f"  Step size = {gamma}")
        print(f"  New X = {X_new}")
        print(f"  New function value = {f(X_new)}")
        history.append(X_new)
        
        X_ = X_new
        
    return X_, history

# Run optimizations
X_final1, history1 = optimize(5, 4, 0.01)
print(f"Final Result: {X_final1}")

print("++++++++++++++++++++++++++++Smaller Step++++++++++++++++++")

X_final2, history2 = optimize(5, 4, 0.0001)
print(f"Final Result: {X_final2}")

# Create plots
plt.figure(figsize=(15, 5))

# Create contour data
X1, X2 = np.meshgrid(np.linspace(-2, 6, 100), np.linspace(-2, 6, 100))
Z = 100 * (X2 - X1**2)**2 + (1 - X1)**2
levels = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Plot first optimization
plt.subplot(121)
plt.contour(X1, X2, Z, levels=levels)
history1 = np.array(history1)
plt.plot(history1[:,0], history1[:,1], 'ro-', label='Optimization path')
plt.plot(history1[0,0], history1[0,1], 'go', label='Start')
plt.plot(history1[-1,0], history1[-1,1], 'bo', label='End')
for i, point in enumerate(history1):
    plt.annotate(f'{i}', (point[0], point[1]), xytext=(10, 10), textcoords='offset points')
plt.grid(True)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Gradient Descent (γ=0.01)')
plt.legend()

# Plot second optimization
plt.subplot(122)
plt.contour(X1, X2, Z, levels=levels)
history2 = np.array(history2)
plt.plot(history2[:,0], history2[:,1], 'ro-', label='Optimization path')
plt.plot(history2[0,0], history2[0,1], 'go', label='Start')
plt.plot(history2[-1,0], history2[-1,1], 'bo', label='End')
for i, point in enumerate(history2):
    plt.annotate(f'{i}', (point[0], point[1]), xytext=(10, 10), textcoords='offset points')
plt.grid(True)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Gradient Descent (γ=0.0001)')
plt.legend()

plt.tight_layout()
plt.show()