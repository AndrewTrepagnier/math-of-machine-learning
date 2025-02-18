import numpy as np
import matplotlib.pyplot as plt

# First plot: Newton Method
epochs = 10
gamma_init = 0.1
Xn = np.array([0.5, -1])
history_newton = []

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

def inverse_hess(X_):
    df_2_x1 = 800*X_[0]**2 - 400*(X_[1] - X_[0]**2) + 2
    df_2_x2 = 200
    df_2_x1x2 = -400*X_[0]
    hessian = np.array([[df_2_x1, df_2_x1x2],
                       [df_2_x1x2, df_2_x2]])
    return np.linalg.inv(hessian)

# Run Newton Method
history_newton.append(Xn.copy())
print("Running Newton Method:")
print(f"Starting at X0 = {Xn}")

for n in range(epochs):
    print(f"\nEpoch {n+1}:")
    gamma = gamma_init
    
    grad = f_grad(Xn)
    newton_dir = inverse_hess(Xn) @ grad
    
    while True:
        if f(Xn - gamma*newton_dir) <= f(Xn) - (gamma/2)*(grad @ newton_dir):
            Xn = Xn - gamma*newton_dir
            history_newton.append(Xn.copy())
            print(f"  Step accepted, new X = {Xn}")
            break
        else:
            gamma = gamma/2
            print(f"  Reducing gamma to {gamma}")

# Second plot: Gradient Descent
Xn_gd = np.array([0.5, -1])  # Same starting point
history_gd = [Xn_gd.copy()]

print("\nRunning Gradient Descent:")
for n in range(epochs):
    gamma = gamma_init
    while True:
        if f(Xn_gd - gamma*f_grad(Xn_gd)) <= f(Xn_gd) - (gamma/2)*(np.linalg.norm(f_grad(Xn_gd)))**2:
            Xn_gd = Xn_gd - gamma*f_grad(Xn_gd)
            history_gd.append(Xn_gd.copy())
            break
        else:
            gamma = gamma/2

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))

# Create contour data with more levels
x = np.linspace(-2, 6, 100)
y = np.linspace(-2, 6, 100)
X, Y = np.meshgrid(x, y)
Z = 100*(Y - X**2)**2 + (1-X)**2

# Define more contour levels
levels = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]

# Plot Newton Method
history_newton = np.array(history_newton)
ax1.contour(X, Y, Z, levels=levels)
ax1.plot(history_newton[:,0], history_newton[:,1], 'ro-', label='Optimization path')
ax1.plot(history_newton[0,0], history_newton[0,1], 'go', label='Start')
ax1.plot(history_newton[-1,0], history_newton[-1,1], 'bo', label='End')
for i, point in enumerate(history_newton):
    ax1.annotate(f'{i}', (point[0], point[1]), xytext=(10, 10), textcoords='offset points')
ax1.grid(True)
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.set_title('Newton Method with BTLS')
ax1.legend()

# Plot Gradient Descent
history_gd = np.array(history_gd)
ax2.contour(X, Y, Z, levels=levels)
ax2.plot(history_gd[:,0], history_gd[:,1], 'ro-', label='Optimization path')
ax2.plot(history_gd[0,0], history_gd[0,1], 'go', label='Start')
ax2.plot(history_gd[-1,0], history_gd[-1,1], 'bo', label='End')
for i, point in enumerate(history_gd):
    ax2.annotate(f'{i}', (point[0], point[1]), xytext=(10, 10), textcoords='offset points')
ax2.grid(True)
ax2.set_xlabel('x₁')
ax2.set_ylabel('x₂')
ax2.set_title('Gradient Descent with BTLS')
ax2.legend()

plt.show()

print(f"\nNewton Method final result: {Xn}")
print(f"Gradient Descent final result: {Xn_gd}")

