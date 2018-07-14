import  numpy as  np
import math

def sigmod(z):
    s = 1 / (1 + math.exp(-z))
    return s

print(sigmod(3))

x = [1, 2, 3]
print(sigmod(x))