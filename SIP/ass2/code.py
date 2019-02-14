import matplotlib.pyplot as plt
import skimage.io as ski
from skimage.io import imread
from pylab import ginput
import numpy as np
import numpy.random as npr
from skimage import img_as_ubyte
from skimage import img_as_uint
from skimage.color import rgb2hsv
from skimage.transform import resize
from skimage.color import rgba2rgb
from skimage.exposure import histogram

image1 = imread("Images/pout.tif")
print(image1.shape)
(a1, a2) = histogram(image1)
print(np.max(image1))
print(np.min(image1))
print(len(a1))
print(len(a2))

print(a1)
print(a2)

def cum_hist(histogram):
  
  return (np.cumsum(histogram[0]), histogram[1])


(y, x) = cum_hist(histogram(image1))
plt.plot(x, y)
plt.ylabel("Frequency")
plt.xlabel("Pixel Intensity")
plt.xlim(0, 255)
plt.ylim(bottom = 0)
plt.legend(["Cumulative Histogram"])
plt.title("Cumulative Histogram of pout.tif")
plt.show()


