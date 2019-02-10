import matplotlib.pyplot as pl
from skimage.io import imread
import pylab as pyl
import numpy as np
import numpy.random as npr

def blend(A, B, w_A, w_B):
  result = np.multiply(w_A, A) + np.multiply(w_B, B)
  return np.round(result).astype('int')

image1 = imread("Images/toycars1.png")
image2 = imread("Images/toycars2.png")
w_b = image1.astype('float')
oneMatrix = image1.astype('float')
oneMatrix.fill(1.0)
pl.figure()
for i in range(5):
  pl.subplot(1, 5, i + 1)
  w_b.fill(i / 4.0)
  w_a = oneMatrix - w_b 
  result = blend(image1, image2, w_a, w_b)
  pl.axis('off')
  pl.title("%s%%, %s%%" % ((1 - i / 4) * 100, i /4 * 100))
  pl.imshow(result)

pl.show()
