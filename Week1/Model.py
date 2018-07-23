import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from Week1.lr_utils import load_dataset

train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()


# train_set_x shape (number of training examples, pixel, pixel, 3)
m_train = train_set_x.shape[0]
m_test = test_set_x.shape[0]
num_px = test_set_x.shape[1]
print(m_train)
print(m_test)
print(num_px)

train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], num_px * num_px * 3).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], num_px * num_px * 3).T
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255
print(test_set_x.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = 1 / m * np.dot(X, (A - Y).T)
