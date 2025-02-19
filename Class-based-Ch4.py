import numpy as np
import matplotlib.pyplot as plt

class RosenbrockOptimizer:
    """
    A class to optimize the Rosenbrock function using various methods:
    - Gradient Descent
    - Gradient Descent with Backtracking Line Search
    - Newton's Method
    - Newton's Method with Diagonal Hessian
    """
    
    def __init__(self, x_start=np.array([0.5, -0.5]), epochs=10):
        """
        Initialize optimizer with starting point and number of epochs
        
        Args:
            x_start (np.array): Starting point [x₁, x₂]
            epochs (int): Number of iterations to run
        """
        self.x_start = x_start
        self.epochs = epochs
        
    def f(self, X_):
        """Rosenbrock function: f(x,y) = 100(y - x²)² + (1-x)²"""
        return 100 * (X_[1] - X_[0]**2)**2 + (1 - X_[0])**2
    
    def f_grad(self, X_):
        """Gradient of Rosenbrock function"""
        df_x1 = -400*X_[0]*(X_[1] - X_[0]**2) - 2*(1 - X_[0])
        df_x2 = 200*(X_[1] - X_[0]**2)
        return np.array([df_x1, df_x2])
    
    def hessian(self, X_):
        """Full Hessian matrix of Rosenbrock function"""
        df_2_x1 = 800*X_[0]**2 - 400*(X_[1] - X_[0]**2) + 2
        df_2_x2 = 200
        df_2_x1x2 = -400*X_[0]
        return np.array([[df_2_x1, df_2_x1x2],
                        [df_2_x1x2, df_2_x2]])
    
    def diagonal_hessian(self, X_):
        """Diagonal approximation of Hessian"""
        df_2_x1 = 800*X_[0]**2 - 400*(X_[1] - X_[0]**2) + 2
        df_2_x2 = 200
        return np.array([[df_2_x1, 0],
                        [0, df_2_x2]])
    
    def gradient_descent(self, gamma=0.01):
        """Basic gradient descent"""
        X_ = self.x_start.copy()
        history = [X_.copy()]
        
        for _ in range(self.epochs):
            X_ = X_ - gamma * self.f_grad(X_)
            history.append(X_.copy())
            
        return history
    
    def gd_with_backtracking(self, gamma=0.01):
        """Gradient descent with backtracking line search"""
        X_ = self.x_start.copy()
        history = [X_.copy()]
        
        for _ in range(self.epochs):
            gamma_t = gamma
            while True:
                grad = self.f_grad(X_)
                if self.f(X_ - gamma_t * grad) <= self.f(X_) - (gamma_t/2) * np.linalg.norm(grad)**2:
                    X_ = X_ - gamma_t * grad
                    history.append(X_.copy())
                    break
                gamma_t = gamma_t / 2
                
        return history
    
    def newton(self, gamma=0.01):
        """Standard Newton's method"""
        X_ = self.x_start.copy()
        history = [X_.copy()]
        
        for _ in range(self.epochs):
            grad = self.f_grad(X_)
            H_inv = np.linalg.inv(self.hessian(X_))
            X_ = X_ - gamma * (H_inv @ grad)
            history.append(X_.copy())
            
        return history
    
    def newton_diagonal(self, gamma=0.01):
        """Newton's method with diagonal Hessian approximation"""
        X_ = self.x_start.copy()
        history = [X_.copy()]
        
        for _ in range(self.epochs):
            grad = self.f_grad(X_)
            H_diag = self.diagonal_hessian(X_)
            X_ = X_ - gamma * np.array([grad[0]/H_diag[0,0], grad[1]/H_diag[1,1]])
            history.append(X_.copy())
            
        return history
    
    def plot_all_methods(self, gamma=0.01):
        """Plot all optimization methods for comparison"""
        # Get histories for all methods
        history_gd = np.array(self.gradient_descent(gamma))
        history_gdls = np.array(self.gd_with_backtracking(gamma))
        history_newton = np.array(self.newton(gamma))
        history_newton_diag = np.array(self.newton_diagonal(gamma))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
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
            (ax3, history_newton, "Newton's Method"),
            (ax4, history_newton_diag, "Newton's with Diagonal H")
        ]:
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

# Example usage:
if __name__ == "__main__":
    optimizer = RosenbrockOptimizer(x_start=np.array([0.5, -0.5]), epochs=10)
    optimizer.plot_all_methods(gamma=0.01) 