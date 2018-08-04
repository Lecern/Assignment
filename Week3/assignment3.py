import numpy as np
import matplotlib.pyplot as plt
from Week3.testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Week3.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), s=40, cmap=plt.cm.Spectral)
# plt.show()

num_examples = X.shape[1]
shape_x = X.shape
shape_y = Y.shape
print("X shape:" + str(shape_x))
print("Y shape:" + str(shape_y))
print("I have m = %d training examples" % num_examples)


# # Train the logistic regression classifier
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)
# # Plot the logistic boundary for logistic regression
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# # plt.show()
# # Print accuracy
# LR_predictions = clf.predict(X, Y)
# print('Accuracy of logistic regression: %d ' % float(
#     (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
#       '% ' + "(percentage of correctly labelled datapoints)")


def layer_sizes(X, Y):
    """
       Arguments:
       X -- input dataset of shape (input size, number of examples)
       Y -- labels of shape (output size, number of examples)

       Returns:
       n_x -- the size of the input layer
       n_h -- the size of the hidden layer
       n_y -- the size of the output layer
       """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


X_access, Y_access = layer_sizes_test_case()
n_x, n_h, n_y = layer_sizes(X_access, Y_access)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)

    W1 = np.random.rand(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


X_access, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_access, parameters)
# Note: we use the mean here just to make sure that your output matches ours.
print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))


def compute_costs(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -1 / m * np.sum(logprobs)

    # use the np.dot and the type of cost is np.array
    # logprobs = np.dot(np.log(A2), Y.T) + np.dot(np.log(1 - A2), (1 - Y.T))
    # cost = -1 / m * logprobs
    cost = np.squeeze(cost)
    assert (isinstance(cost, float))
    return cost


A2, Y_access, parameters = compute_cost_test_case()
cost = compute_costs(A2, Y_access, parameters)
print("cost= " + str(cost))
print(type(cost))


def backward_propagation(paramaters, cache, X, Y):
    m = X.shape[1]
    # First, retrieve W1 and W2 from the dictionary "parameters"
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    # Retrieve also A1 and A2 from the dictionary "cahce"
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2
    dZ2 = np.subtract(A2, Y)
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
