import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)

plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")