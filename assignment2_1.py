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
print(np_sigmoid(np_x))
