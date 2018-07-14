import numpy as np
import math


def sigmod(z):
    s = 1 / (1 + math.exp(-z))
    return s


# one example of math
# print(sigmod(3))

x = [1, 2, 3]


# One reason why we use "numpy" instead of "math" in Deep Learning
# print(sigmod(x))

# one example of np.exp
# print(np.exp(x))

def np_sigmoid(z):
    return 1 / (1 + np.exp(-z))


np_x = np.array([1, 2, 3])
print("sigmoid" + str(np_x) + ":" + str(np_sigmoid(np_x)))


# derivative
def np_sigmoid_derivative(z):
    fx = 1 / (1 + np.exp(-z))
    return fx * (1 - fx)


print("sigmoid_derivative" + str(np_x) + ":" + str(np_sigmoid_derivative(np_x)))


# np.shape np.reshape
def image2vector(image):
    """

    :param image: a numpy array of shape(length height depth)
    :return: a vector of shape(length*height*depth,1)
    """
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)


np_matrix = np.array([[[0.67826139, 0.29380381],
               [0.90714982, 0.52835647],
               [0.4215251, 0.45017551]],

              [[0.92814219, 0.96677647],
               [0.85304703, 0.52351845],
               [0.19981397, 0.27417313]],

              [[0.60659855, 0.00533165],
               [0.10820313, 0.49978937],
               [0.34144279, 0.94630077]]])
print("reshape:" + str(image2vector(np_matrix)))

