import numpy as np


x1 = np.array([[1, 2, 3], [4, 5, 6]])
x2 = np.array([[1, 2], [3, 4], [5, 6]])
x3 = np.array([[1, 2], [3, 4]])
x4 = np.array([[5, 6], [7, 8]])
# 矩阵乘法
dot = np.dot(x1, x2)
# 笛卡儿积(形状相同)
mul = np.multiply(x3, x4)
print("Dot product =" + str(dot))
print("Element wise product=" + str(mul))
print(str(np.sum(mul)))
