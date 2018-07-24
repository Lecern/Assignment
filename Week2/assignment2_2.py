import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from Week2.lr_utils import load_dataset
MAX_PIXEL = 255


# Loading the data (cat/non cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# Example of a picture
index = 158  # free to set
plt.imshow(train_set_x_orig[index])
plt.show()
print("y= " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode('utf-8')
      + "' picture")


m_train = train_set_x_orig.shape[0]  # m_train: number of training examples
m_test = test_set_x_orig.shape[0]  # m_test: number of test examples
num_px = test_set_x_orig.shape[1]  # num_px: =height =width of a training image
print("Number of training examples: m_train= " + str(m_train))
print("Number of test examples: m_test= " + str(m_test))
print("Height/Width of each image: num_px= " + str(num_px))
print("Each image is of size (" + str(num_px) + "," + str(num_px) + ",3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))


# Reshape the train and test examples
# There is a trick
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))


# Standardize
train_set_x = train_set_x_flatten / MAX_PIXEL
test_set_x = test_set_x_flatten / MAX_PIXEL

"""
What you need to remember:

Common steps for pre-processing a new dataset are:

    1.Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    2.Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
    3."Standardize" the data
"""


# define sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


print("sigmoid([0,2])= " + str(sigmoid(np.array([0,2]))))


# define initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape(dim,1) for w and initializes b to 0
    :param dim: size of the w vector we want(or number of parameters in this case)
    :return w: initialized vector of shape(dim, 1)
            b: initialized scale(corresponds to the bias)
    """
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w,b


dim = 2
w, b = initialize_with_zeros(dim)
print("w=" + str(w))
print("b=" + str(b))


# define propagate
def propagate(w, b, X, Y):
    """
    Implement cost function and its gradient for the propagation
    :param w: weights, a numpy array of size(num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size(num_px * num_px * 3, number of examples)
    :param Y: true label vector(containing 0 if non-cat, 1 if cat) of size(1, number of examples)
    :return cost: negative log-likelihood cost fot logistic regression
            dw: gradient of the loss with the respect to w, thus same shape as w
            db: gradient of the loss with the respect to b, thus same shape as b
    """
    # X为行向量, 此处m为X的列数, 即样本个数
    m = X.shape[1]
    # 以下计算注意区分矩阵乘法和数字乘法
    # FROM X to cost
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost

    # backward propagation(to find grad)
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {'dw': dw, 'db': db}
    return grads, cost


# 这里w为3*1 X为4*2 Y为1*2
w, b, X, Y = np.array([[1], [2], [3]]), 2, np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads['dw']))
print("db = " + str(grads['db']))
print("cost =" + str(cost))


def optimize(w, b, X, Y, number_iterations, learning_rate, print_cost = False):
    """
    The function optimizes w and b by running a gradient descent algorithm
    :param w: weights, a numpy array of size(num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: date of shape(num_px * num_px * 3, number of examples)
    :param Y: true "label" vectors(containing 0 if non-cat, 1 if cat) of shape(1, number of examples)
    :param number_iterations: number of iteration of the optimization loop
    :param learning_rate: learning rate of the gradient descent update rule
    :param print_cost: True to print the loss every 100 steps
    :return: params: dictionary containing the weights w and bias b
             grads: dictionary containing the gradient of the weights and bias with respect to the cost function
             costs: list of all the costs computed during optimization, this will be used to plot the curve

     Tips:
        1) Calculate the cost and the gradient for the current parameters. Use propagate() function.
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    for i in range(number_iterations):
        # cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # retrieve derivatives from goals
        dw = grads['dw']
        db = grads['db']

        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # record cost
        if i % 100 == 0:
            costs.append(cost)

        # print costs every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}
    return params, grads, costs


params, grads, costs = optimize(w, b, X, Y, number_iterations=101, learning_rate=0.009, print_cost=False)
print("w= " + str(params['w']))
print("b= " + str(params['b']))
print("dw= " + str(grads['dw']))
print("db= " + str(grads['db']))
print("costs= " + str(costs))


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters(w, b)
    :param w: weights, a numpy array of size(num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size(num_px * num_px * 3, number of examples)
    :return Y_prediction: a numpy array(vector) containing all predictions(0/1) from the examples in X
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute the vector "A" predicting probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # Convert probabilities A[0, i] to actual predictions p[0, i]
        if (A[0, i]) < 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    assert (Y_prediction.shape == (1,m))
    return Y_prediction


print("Prediction= " + str(predict(w, b, X)))
"""
    What to remember
    -Initialize (w, b)
    -Optimize the loss iteratively to learn parameters(w, b):
        -computing the cost and its gradient
        -updating the parameters using gradient descent
    -Use the learned(w, b) to predict the labels for a given set of examples 
"""


def model(X_train, Y_train, X_test, Y_test, num_iterations=200, learning_rate=0.5, print_cost=False):
    """
    Build the logistic regression model by calling the functions you've implemented previously
    :param X_train: training set represented by a numpy array of shape(num_px * num_px * 3, m_train)
    :param Y_train: training labels represented by a numpy array of shape(1, m_train)
    :param X_test: test set represented by a numpy array of shape(num_px * num_px * 3, m_train)
    :param Y_test: test labels represented by a numpy array of shape(1, m_train)
    :param num_iterations: hyperparameter representing the number of iterations to optimize the parameters
    :param learning_rate: hyperparameter representing the learning rate used in the update rule of optimize()
    :param print_cost: Set to true print the cost evert 100 iteration
    :return d: dictionary containing information about the model
    """
    # Initialize parameters with zeros
    w,b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    params, grads, cost= optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = params['w']
    b = params['b']

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print test/train error
    print("test accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    print("train accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    d = {
        "cost": cost,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iteration": num_iterations
    }
    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)


index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
plt.show()
print("y= " + str(test_set_y[0, index]) + " your predicted that it is a \"" +
      classes[int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture")


# Plot learning curve(with costs)
costs = np.squeeze(d['cost'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iteration (per hundreds)')
plt.title('Learning rate:' + str(d["learning_rate"]))
plt.show()


# Choice of learning rate
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("Learning rate is " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000,
          learning_rate=i, print_cost=True)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["cost"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel("cost")
plt.xlabel("iterations")

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
