import h5py
import numpy 
import numpy as np
import os, random
from skimage import color
from skimage import io
import matplotlib.pyplot as plt

f = np.array(color.rgb2gray(io.imread('test_images/seg/1003_3_seg.png')))
plt.imshow(f)
plt.show()
