import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from Week1.lr_utils import load_dataset


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
print("train_set_x shape:" + str(train_set_x_orig.shape))
print("train_set_y shape:" + str(train_set_y.shape))
print("test_set_x shape:" + str(test_set_x_orig.shape))
print("test_set_y shape:" + str(test_set_y.shape))
