import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from Week1.lr_utils import load_dataset
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
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {'dw': dw, 'db': db}
    return grads, cost


w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])  # 这里w为2*1 X为2*2 Y为1*2
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads['dw']))
print("db = " + str(grads['db']))
print("cost =" + str(cost))
