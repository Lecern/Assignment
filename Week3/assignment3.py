import numpy as np
import matplotlib.pyplot as plt
from Week3.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Week3.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y.reshape(400), s=40, cmap=plt.cm.Spectral)


num_examples = X.shape[1]
x_shape = X.shape
y_shape = Y.shape
print("X shape:" + str(x_shape))
print("Y shape:" + str(y_shape))
print("I have m = %d training examples" % num_examples)


clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)