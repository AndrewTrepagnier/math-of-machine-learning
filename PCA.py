import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score





df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/' 'machine-learning-databases/wine/wine.data', header=None)

"""
Seperate the wine dataset into seperate training and testing datsets
"""
X , y = df_wine.iloc[:,1:].values , df_wine.iloc[:,0].values #how does this splice the dataset exactly?

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# how can we standardize any feature set?

sc = StandardScaler()

X_train_std, X_test_std = sc.fit_transform(X_train), sc.transform(X_test)

""" Why is it mandatory that the dataset is standardized before beginning any PCA transformations"""

# remember that d is the number of the dimensions of the dataset

#Next we compute the covariance matrix

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(f" The eigenvectors of the covariance matrix is {eigen_vecs}")
print(f"The eigenvalues of the covariance matrix is {eigen_vals}")


"""Total and Explained Variance"""

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in  sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

#Plotting

plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
"""
The resulting plot indicates that the first principal component alone accounts for 40% of the variance.
In the next steps, we will sort the eigenpairs by desending order of the eigenvalues, construct a projection matrix
from the selected eigenvecotrs, and use the projection matrix to transform the data into a lower dimensional space
"""

# Start by sorting the eigenpairs by decreasing order of the eigenvectors

#How do you actually sort a tuple of matrices in order if they consist of different values??

# make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals)) ] # study how a for loop is implemented within an array declaration
eigen_pairs.sort(key=lambda k:k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W: \n', w)

X_train_std[0].dot (w)
X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers= ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


"""SKlearn implementation of PCA"""

def plot_decision_regions(X, y, classifier, resolution=0.02):
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
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                   y=X[y == cl, 1],
                   alpha=0.6, 
                   c=colors[idx],
                   marker=markers[idx],
                   label=cl,
                   edgecolor='black')
        
pca = PCA(n_components=2)
lr = OneVsRestClassifier(LogisticRegression(random_state=1, solver='lbfgs'))

X_train_pca =pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_

"""Finding optimal number of components"""

# Store accuracies for different numbers of components
accuracies = []
n_components_range = range(1, 14)  # 1 to 13 components

for n in n_components_range:
    # Create and fit PCA
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    
    # Train and evaluate classifier
    lr = OneVsRestClassifier(LogisticRegression(random_state=1, solver='lbfgs'))
    lr.fit(X_train_pca, y_train)
    y_pred = lr.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    print(f"n_components={n}, Accuracy={acc:.4f}, Cumulative explained variance={np.sum(pca.explained_variance_ratio_):.4f}")

# Find optimal number of components
k_star = n_components_range[np.argmax(accuracies)]
print(f"\nOptimal number of components (k*) = {k_star}")

# Plot accuracy vs number of components
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(n_components_range, accuracies, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Classification Accuracy')
plt.title('Accuracy vs Number of Components')

# Plot cumulative explained variance
pca = PCA(n_components=None)  # None means keep all components
X_train_pca = pca.fit_transform(X_train_std)
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cum_var_exp) + 1), cum_var_exp, 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.axvline(x=k_star, color='g', linestyle='--', label=f'k*={k_star}')
plt.legend()

plt.tight_layout()
plt.show()

# Print the cumulative explained variance for k*
print(f"Cumulative explained variance at k*={k_star}: {cum_var_exp[k_star-1]:.4f}")

# Use the optimal number of components for the final model
pca = PCA(n_components=k_star)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = OneVsRestClassifier(LogisticRegression(random_state=1, solver='lbfgs'))
lr.fit(X_train_pca, y_train)

# Plot decision regions with optimal number of components
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Decision Regions (k*={k_star} components)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

