import numpy as np
import matplotlib.pyplot as plt

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

