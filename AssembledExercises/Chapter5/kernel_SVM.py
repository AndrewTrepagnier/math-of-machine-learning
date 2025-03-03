from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from inseperable_dataset import X_xor, y_xor

# Load iris data
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]  # sepal length and petal length
y = iris.target  # all three classes

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# Standardization
X_train_std = np.copy(X_train)
X_train_std[:,0] = (X_train[:,0] - X_train[:,0].mean()) / X_train[:,0].std()
X_train_std[:,1] = (X_train[:,1] - X_train[:,1].mean()) / X_train[:,1].std()

X_test_std = np.copy(X_test)
X_test_std[:,0] = (X_test[:,0] - X_train[:,0].mean()) / X_train[:,0].std()  # Use training mean/std
X_test_std[:,1] = (X_test[:,1] - X_train[:,1].mean()) / X_train[:,1].std()

# Combine for plotting
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
test_idx = range(len(y_train), len(y_combined))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                   y=X[y == cl, 1],
                   alpha=0.8, 
                   c=colors[idx],
                   marker=markers[idx],
                   label=cl,
                   edgecolor='black')
        
    # Highlight test samples if provided
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                   c='none',
                   edgecolor='black',
                   alpha=1.0,
                   linewidth=1,
                   marker='o',
                   s=100,
                   label='test set')

# Train SVM with linear kernel
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

# Create figure for SVM
plt.figure(figsize=(8, 6))
plot_decision_regions(X_combined_std, y_combined, 
                     classifier=svm, 
                     test_idx=test_idx)
plt.title('SVM with Linear Kernel')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# After plotting our custom models, add sklearn's implementation
plt.figure(figsize=(8, 6))

# Create and train sklearn's logistic regression
lr = LogisticRegression(C=100.0, random_state=1,
                       solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)  # Using our standardized training data

# Plot decision regions
plot_decision_regions(X_combined_std, y_combined, 
                     classifier=lr,
                     test_idx=test_idx)
plt.title('Scikit-learn Logistic Regression')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# After the logistic regression plot, add the XOR example with RBF kernel

# Create a new figure for XOR dataset
plt.figure(figsize=(8, 6))

# Train SVM with RBF kernel on XOR data
svm_rbf = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10)
svm_rbf.fit(X_xor, y_xor)

# Plot decision regions
plot_decision_regions(X_xor, y_xor, classifier=svm_rbf)
plt.title('SVM with RBF Kernel on XOR Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()