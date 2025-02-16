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
def f(x):
    return x**2

def grad_f(x):
    return 2*x


gamma = .01  
x = 5.0  
iterations = 10
history = [x]  

# GD Algorithm
for i in range(iterations):
    xnew = x -   gamma*grad_f(x)
    x = xnew
    history.append(x)

# Visualization
x_vals = np.linspace(-6, 6, 100)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label="f(x) = x^2")
plt.scatter(history, [f(i) for i in history], color='red', marker='o', label="Gradient descent path")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("Gradient Descent on f(x) = x^2 and step size of %f" %gamma)
plt.show()
