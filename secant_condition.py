import numpy as np
import matplotlib.pyplot as plt

def f(X_): 
    result = 100 * (X_[1] - X_[0]**2)**2 + (1 - X_[0])**2
    print(f"f(X) = {result}")  
    return result

def f_grad(X_): 
    df_x1 = -400*X_[0]*(X_[1] - X_[0]**2) -2 * (1- X_[0])
    df_x2 = 200 * (X_[1] - X_[0]**2) 
    grad = np.array([df_x1, df_x2])
    print(f"gradient = {grad}")
    return grad

def inverse_hess(X_):
    df_2_x1 = 800*X_[0]**2 - 400*(X_[1] - X_[0]**2) + 2
    df_2_x2 = 200
    df_2_x1x2 = -400*X_[0]
    hessian = np.array([[df_2_x1, df_2_x1x2],
                       [df_2_x1x2, df_2_x2]])
    return np.linalg.inv(hessian)

#++++++++++++++++++++++++++++++++++++++++++++++ Using Current and Next Steps to Find S_n and y_n
# 1. Start with ONE initial point
Xn = np.array([0.5, -0.5])
grad_n = f_grad(Xn)  # Current gradient

# 2. Take a step using Newton's method instead of gradient descent
newton_dir = inverse_hess(Xn) @ grad_n  # Newton direction
gamma = 0.01
Xn_next = Xn - gamma * newton_dir  # Note the minus sign

# 3. Now we can compute s_n and y_n
s_n = Xn_next - Xn                    # The step we took
grad_next = f_grad(Xn_next)           # Gradient at new point
y_n = grad_next - grad_n   

# Start with true Hessian and its inverse
H_current = inverse_hess(Xn)  # Get actual inverse Hessian at current point
B_current = np.linalg.inv(H_current)  # Get actual Hessian


print("\nVerifying initial H*B = I:")
print(H_current @ B_current)
print("Difference from identity:", np.linalg.norm(H_current @ B_current - np.eye(2)))

# Now compute updates
term1 = np.outer(y_n, y_n) / (y_n @ s_n)
term2 = (H_current @ s_n) @ (H_current @ s_n).T / (s_n @ H_current @ s_n)
H_next = H_current + term1 - term2

term1_B = np.eye(2) - np.outer(s_n, y_n.T) / (y_n @ s_n)
term2_B = np.eye(2) - np.outer(y_n, s_n.T) / (y_n @ s_n)
term3_B = np.outer(s_n, s_n.T) / (y_n @ s_n)
B_next = term1_B @ B_current @ term2_B + term3_B

# Verify H_next and B_next are still inverses
print("\nVerifying H_next * B_next = I:")
print(H_next @ B_next)
print("Difference from identity:", np.linalg.norm(H_next @ B_next - np.eye(2)))

print("\nH_next (Inverse Hessian):")
print(H_next)
print("\nB_next (Direct Hessian):")
print(B_next)


print("\nVerifying Secant Conditions:")
print("y_n =", y_n)
print("H_next @ s_n =", H_next @ s_n)
print("B_next @ s_n =", B_next @ s_n)
print("\nDifferences (should be close to zero):")
print("H difference:", np.linalg.norm(y_n - H_next @ s_n))
print("B difference:", np.linalg.norm(y_n - B_next @ s_n))


print("\nVerifying H_next * B_next = Identity:")
HB_product = H_next @ B_next
print("H_next @ B_next =")
print(HB_product)


I = np.eye(2)
print("\nDifference from identity (should be close to zero):")
print(np.linalg.norm(HB_product - I))


print("\nB_next @ H_next =")
print(B_next @ H_next)
print("\nDifference from identity (should be close to zero):")
print(np.linalg.norm(B_next @ H_next - I))

#================== Continue BFGS Iteration ==================
# Store first iteration results
history_of_iterations = [Xn.copy(), Xn_next.copy()]
H_current = H_next 


for i in range(9):  
    Xn = history_of_iterations[-1] 
    grad_n = f_grad(Xn)
    
    search_dir = -np.linalg.solve(H_current, grad_n)
  
    Xn_next = Xn + gamma * search_dir
    
    # Compute s_n and y_n for update
    s_n = Xn_next - Xn
    grad_next = f_grad(Xn_next)
    y_n = grad_next - grad_n
    
    
    term1 = np.outer(y_n, y_n) / (y_n @ s_n)
    term2 = (H_current @ s_n) @ (H_current @ s_n).T / (s_n @ H_current @ s_n)
    H_current = H_current + term1 - term2
    
    
    history_of_iterations.append(Xn_next.copy())
    print(f"\nIteration {i+2}:")
    print(f"Point: {Xn_next}")
    print(f"Function value: {f(Xn_next)}")


history_ = np.array(history_of_iterations)


plt.figure(figsize=(10,8))


x = np.linspace(-2, 6, 100) 
y = np.linspace(-2, 6, 100)  
X, Y = np.meshgrid(x, y)
Z = 100*(Y - X**2)**2 + (1-X)**2


levels = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
plt.contour(X, Y, Z, levels=levels)

plt.plot(history_[:,0], history_[:,1], 'ro-', label='Optimization path')
plt.plot(history_[0,0], history_[0,1], 'go', label='Start')
plt.plot(history_[-1,0], history_[-1,1], 'bo', label='End')

for i, point in enumerate(history_):
    plt.annotate(f'{i}', (point[0], point[1]), xytext=(10, 10), textcoords='offset points')

plt.grid(True)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('BFGS Method on Rosenbrock Function')
plt.legend()
plt.show()