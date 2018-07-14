import numpy as np
import math


# original sigmoid function
def sigmoid(z):
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
    """
    取x的导数
    :param z:  A scalar or numpy array
    :return:  Your computed gradient.
    """
    fx = 1 / (1 + np.exp(-z))
    return fx * (1 - fx)


print("sigmoid_derivative" + str(np_x) + ":" + str(np_sigmoid_derivative(np_x)))


# np.shape np.reshape
def image2vector(image):
    """
    这里是将3*3*2矩阵降维
    :param image: a numpy array of shape(length height depth)
    :return: a vector of shape(length*height*depth,1)
    """
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)


# 3*3*2数组
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


def normalize_rows(x):
    """
    归一化 矩阵的各个元素除以模
    :param x: a numpy matrix of shape (n, m)
    :return: the normalized (by rows) numpy matrix
    """
    """
    x_norm=np.linalg.norm(x, ord=None, axis=None, keepdims=False)
    ①x: 表示矩阵（也可以是一维）
    ②ord：范数类型
    ③axis：处理类型 axis=1表示按行向量处理，求多个行向量的范数
    ④keepdims：是否保持矩阵的二维特性
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    print("-------------x_norm=" + str(x_norm) + "-------------")
    return x / x_norm


np_array = np.array([[0, 3, 4], [1, 6, 4]])
print("normalize_rows:" + str(normalize_rows(np_array)))


def softmax(x):
    """
    模拟softmax
    :param x: a numpy matrix of shape (n, m)
    :return: a numpy matrix equal to the softmax of x, of shape (n, m)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / x_sum


np_array2 = np.array([[9, 2, 5, 0, 0],
    [7, 5, 0, 0, 0]])
print("softmax:" + str(softmax(np_array2)))
