import numpy as np
import time

"""
    ---------------------vectorization start---------------------
"""
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

# Classic dot product implementation #
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print("dot=" + str(dot) + "\n -----------Dot Computation time :" + str(1000*(toc-tic)) + "ms-----------")

# Classic outer product implementation #
tic = time.process_time()
outer = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i] * x2[j]
toc = time.process_time()
print("outer=" + str(outer) + "\n -----------Outer Computation time :" + str(1000*(toc-tic)) + "ms-----------")

# Classic elementwise implementation #
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
toc = time.process_time()
print("elementwise multiply=" + str(mul) + "\n -----------Element multiply Computation time :" + str(1000*(toc-tic))
      + "ms-----------")

# TODO: What's general dot product
# Classic general dot product(?) implementation #
W = np.random.rand(3, len(x1))  # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j] * x1[j]
toc = time.process_time()
print("gdot=" + str(gdot) + "\n -----------Gdot Computation time :" + str(1000*(toc-tic)) + "ms-----------")

for i in range(10):
    for j in range(160):
        print("*", end="")
    print("")

# Vectorized dot product #
tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()
print("dot=" + str(dot) + "\n -----------Dot Computation time :" + str(1000*(toc-tic)) + "ms-----------")

# Vectorized outer product #
tic = time.process_time()
outer = np.outer(x1, x2)
toc = time.process_time()
print("outer=" + str(outer) + "\n -----------Outer Computation time :" + str(1000*(toc-tic)) + "ms-----------")

# Vectorized element wise implementation #
tic = time.process_time()
mul = np.multiply(x1, x2)
toc = time.process_time()
print("elementwise multiply=" + str(mul) + "\n -----------Element multiply Computation time :" + str(1000*(toc-tic))
      + "ms-----------")

# Vectorized general dot product(?) implementation #
tic = time.process_time()
gdot = np.dot(W, x1)
toc = time.process_time()
print("gdot=" + str(gdot) + "\n -----------Gdot Computation time :" + str(1000*(toc-tic)) + "ms-----------")
"""
    ---------------------vectorization end---------------------
"""