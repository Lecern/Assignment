import numpy as np

x1 = np.array([[1, 2, 3], [4, 5, 6]])  # 2*3
x2 = np.array([[1, 2], [3, 4], [5, 6]])  # 3*2
x3 = np.array([[1, 2], [3, 4]])
x4 = np.array([[5, 6], [7, 8]])
# 矩阵乘法
dot = np.dot(x1, x2)
dot2 = np.dot(x2, x1)
# 笛卡儿积(形状相同)
mul = np.multiply(x3, x4)
print("Dot product =" + str(dot))
print("Element wise product=" + str(mul))
print(str(np.sum(mul)))
print("dot2 = " + str(dot2))

hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
e = enumerate(hidden_layer_sizes)

x5 = np.array([[2]])
print(x5.shape)