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
    """
    Returns diagonal approximation of Hessian
    """
    df_2_x1 = 800*X_[0]**2 - 400*(X_[1] - X_[0]**2) + 2
    df_2_x2 = 200
    diag = np.array([[df_2_x1, 0],
                     [0, df_2_x2]])
    return diag

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

def Newton_method(X_, gamma_parameter):
    """
    Standard Newton's method without line search
    X_{n+1} = X_n - γ * H^(-1)∇f
    """
    stored_Xs = [X_.copy()]  # Store initial point
    
    for _ in range(epochs):
        grad = f_grad(X_)
        H_inv = _inverse_hess_(X_)
        X_ = X_ - gamma_parameter * (H_inv @ grad)
        stored_Xs.append(X_.copy())
        
    return stored_Xs

def Newton_with_diagonal_H(X_, gamma_parameter):
    """
    Newton's method with diagonal Hessian approximation
    """
    #  Newton's method using diagonal Hessian approximation
    
    # Args:
    #     X_: Starting point (numpy array)
    #     gamma_parameter: Step size
    # Returns:
    #     list: History of points visited
    stored_Xs = [X_.copy()]
    
    for _ in range(epochs):
        grad = f_grad(X_)
        H_diag = approximated_hess_(X_)
        # For diagonal H, we should divide each gradient component by corresponding diagonal element
        X_ = X_ - gamma_parameter * np.array([grad[0]/H_diag[0,0], grad[1]/H_diag[1,1]])
        stored_Xs.append(X_.copy())
    
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

#+++++++++++++++++++++++++++++++++Ellipsoid Definitions+++++++++++++++++++++++++++++++

def ellipsoid(a, b, x, y, h, k):
    return ((x - h)**2) / a**2 + ((y - k)**2) / b**2

def grad_ellipse(a, b, x, y, h, k):
    wrtx = (2*(x - h)) / a**2
    wrty = (2*(y - k)) / b**2
    return np.array([wrtx, wrty])

def double_grad_ellipse(a, b, x, y, h, k):
    wrtx_2 = 2/a**2
    wrty_2 = 2/b**2
    return np.array([[wrtx_2, 0], [0, wrty_2]])


#+++++++++++++++++++++++++++++++++++INITIALIZE+++++++++++++++++++++++++++++++++++++++++
X_ = [0.5, -0.5] # Starting point we will optimize FROM

#Carefully select one of these as your optimize argument if you want effective guess of gamma or not
gam = 0.001 # appropriate starting step size
gam_effect = estimate_gamma(X_)

epochs = 20 # keep this consistent across each problem tested
stored_Xs = [X_] # Placing starting point in the storage array


#++++++++++++++++++++++++++++++++++ELLIPSOID PARAMTERS AND CALL ON++++++++++++++++++++++++++++++++++
# Parameters
a, b = 2, 1  
h, k = 0, 0  

# Create grid
x_val = np.linspace(-4, 4, 1000)
y_val = np.linspace(-4, 4, 1000)
X, Y = np.meshgrid(x_val, y_val)

# Plot ellipsoid with multiple level curves
plt.figure(figsize=(10,8))
plt.contour(X, Y, ellipsoid(a,b,X,Y,h,k), levels=[0.25, 0.5, 1, 2, 4], 
           colors=['blue', 'green', 'red', 'purple', 'orange'])

test_points = [
    (2, 0),   # right
    (-2, 0),  # left
    (0, 1),   # top
    (0, -1),  # bottom
    (1, 0.7)  # arbitrary point
]

for x, y in test_points:
    # Calculate gradient and Hessian
    grad = grad_ellipse(a, b, x, y, h, k)
    hess = double_grad_ellipse(a, b, x, y, h, k)
    
    # Calculate Newton direction
    newton_dir = -np.linalg.inv(hess) @ grad
    
    # Plot point
    plt.plot(x, y, 'ko')  # black dot for point
    
    # Plot gradient direction (blue)
    plt.arrow(x, y, grad[0]*0.2, grad[1]*0.2, 
             head_width=0.1, head_length=0.1, fc='b', ec='b', 
             label='Gradient' if (x,y)==test_points[0] else None)
    
    # Plot Newton direction (red)
    plt.arrow(x, y, newton_dir[0]*0.2, newton_dir[1]*0.2, 
             head_width=0.1, head_length=0.1, fc='r', ec='r',
             label='Newton Direction' if (x,y)==test_points[0] else None)

plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ellipsoid Level Curves with Gradient (blue) and Newton (red) Directions')
plt.axis('equal')
plt.legend()
plt.show()





#+++++++++++++++++++++++++++++++++++PROBLEM 1 , 3

def plot_optimization_methods():
    # Use existing initializations
    X_start = np.array(X_)  # Your starting point [0.5, -0.5]
    
    # Run each method using your stored_Xs
    # 1. Gradient Descent
    X_gd = X_start.copy()
    stored_Xs = [X_gd]
    X_final, history_gd = _gradient_descent_(X_gd, gam)
    
    # 2. GD with Backtracking Line Search - Fix initial point storage
    X_gdls = X_start.copy()
    history_gdls = [X_gdls.copy()]  # Store initial point
    history_gdls.extend(GD_with_BTLS(X_gdls, gam_effect))
    
    # 3. Newton with Backtracking Line Search
    X_nls = X_start.copy()
    history_nls = Newton_method(X_nls, 0.1)
    
    # 4. Newton with Diagonal Hessian
    X_nd = X_start.copy()
    history_nd = Newton_with_diagonal_H(X_nd, 0.1)
    
    # Create subplots with adjusted size and spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust spacing
    
    # Create contour data
    x = np.linspace(-2, 6, 100)
    y = np.linspace(-2, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = 100*(Y - X**2)**2 + (1-X)**2
    levels = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    # Plot each method
    for ax, history, title in [
        (ax1, history_gd, 'Gradient Descent'),
        (ax2, history_gdls, 'GD with Line Search'),
        (ax3, history_nls, "Newton's with Line Search"),
        (ax4, history_nd, "Newton's with Diagonal H")
    ]:
        history = np.array(history)
        ax.contour(X, Y, Z, levels=levels)
        ax.plot(history[:,0], history[:,1], 'ro-', label='Path')
        ax.plot(history[0,0], history[0,1], 'go', label='Start')
        ax.plot(history[-1,0], history[-1,1], 'bo', label='End')
        
        for i, point in enumerate(history):
            ax.annotate(f'{i}', (point[0], point[1]), xytext=(10, 10), 
                       textcoords='offset points')
        
        ax.grid(True)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(title)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_optimization_methods()

#+++++++++++++++++++++++++++++++++++PLOTTING+++++++++++++++++++++++++++++++++++++++++

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Create contour data once
x = np.linspace(-2, 6, 100)
y = np.linspace(-2, 6, 100)
X, Y = np.meshgrid(x, y)
Z = 100*(Y - X**2)**2 + (1-X)**2
levels = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# 1. Gradient Descent Plot
X_gd = np.array(X_)
stored_Xs = [X_gd]
X_final, history_gd = _gradient_descent_(X_gd, gam)
history_gd = np.array(history_gd)
ax1.contour(X, Y, Z, levels=levels)
ax1.plot(history_gd[:,0], history_gd[:,1], 'ro-', label='Path')
ax1.plot(history_gd[0,0], history_gd[0,1], 'go', label='Start')
ax1.plot(history_gd[-1,0], history_gd[-1,1], 'bo', label='End')
ax1.set_title('Gradient Descent')
ax1.grid(True)
ax1.legend()

# 2. GD with Line Search Plot
X_gdls = np.array(X_)
history_gdls = [X_gdls]
history_gdls.extend(GD_with_BTLS(X_gdls, gam))
history_gdls = np.array(history_gdls)
ax2.contour(X, Y, Z, levels=levels)
ax2.plot(history_gdls[:,0], history_gdls[:,1], 'ro-', label='Path')
ax2.plot(history_gdls[0,0], history_gdls[0,1], 'go', label='Start')
ax2.plot(history_gdls[-1,0], history_gdls[-1,1], 'bo', label='End')
ax2.set_title('GD with Line Search')
ax2.grid(True)
ax2.legend()

# 3. Newton with Line Search Plot
X_nls = np.array(X_)
history_nls = Newton_method(X_nls, 0.1)
history_nls = np.array(history_nls)
ax3.contour(X, Y, Z, levels=levels)
ax3.plot(history_nls[:,0], history_nls[:,1], 'ro-', label='Path')
ax3.plot(history_nls[0,0], history_nls[0,1], 'go', label='Start')
ax3.plot(history_nls[-1,0], history_nls[-1,1], 'bo', label='End')
ax3.set_title("Newton's with Line Search")
ax3.grid(True)
ax3.legend()

# 4. Newton with Diagonal H Plot
X_nd = np.array(X_)
history_nd = Newton_with_diagonal_H(X_nd, 0.1)
history_nd = np.array(history_nd)
ax4.contour(X, Y, Z, levels=levels)
ax4.plot(history_nd[:,0], history_nd[:,1], 'ro-', label='Path')
ax4.plot(history_nd[0,0], history_nd[0,1], 'go', label='Start')
ax4.plot(history_nd[-1,0], history_nd[-1,1], 'bo', label='End')
ax4.set_title("Newton's with Diagonal H")
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.show()
