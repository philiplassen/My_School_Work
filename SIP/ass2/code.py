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
image2 = imread("Images/cameraman.tif")
image3 = imread("Images/cell.tif")
def cum_hist(histogram):
  """Takes a tuple histogram = (bin, counts) and returns
  the cumulative histogram with integer bins from 0 to 255"""
  (init_counts, init_bins) = histogram
  if isinstance(init_bins[0], np.float64):
    init_bins = np.round(init_bins * 255)
    init_bins = init_bins.astype(int)
  counts = [0 for i in range(256)]
  bins = [i for i in range(256)]
  total = sum(init_counts)
  for i in range(len(init_bins)): 
    counts[init_bins[i]] = float(init_counts[i]) / total
  return (np.round(np.cumsum(counts), 10) , bins)
    
"""
(y, x) = cum_hist(histogram(image1))
plt.plot(x, y)
plt.ylabel("Frequency")
plt.xlabel("Pixel Intensity")
plt.xlim(0, 255)
plt.ylim(bottom = 0)
plt.legend(["Cumulative Histogram"])
plt.title("Cumulative Histogram of pout.tif")
plt.show()
"""
def C(I, CDF):
  """ Takes a Grey Scale Image and a
  CDF and returns their function composition
  Assumes the CDF is an array of length 256"""
  if isinstance(I[0,0], np.float64):
    I = np.round(255 * I).astype(int)
  CI = [[CDF[val] for val in row] for row in I]
  return CI / max(CDF)

"""
(y, x) = cum_hist(histogram(image1))
plt.subplot(1, 2, 1)
plt.imshow(C(image1, y), cmap = 'gray')
plt.subplot(1, 2, 2)
plt.imshow(image1, cmap = 'gray')
plt.show()
"""

def cdf_inverse(l, cdf):
  #print(l)
  return min([s for s in range(256) if cdf[s] >= l])
"""  
(y, x) = cum_hist(histogram(image1))
print(cdf_inverse(.999, y))
print(np.min(image1))
print(np.max(image1))
"""

def histogram_matching(im1, c1, c2):
  temp = C(im1, c1)
  result = [[cdf_inverse(val, c2) for val in row] for row in temp]
  return result


"""
(vals1, bins1) = cum_hist(histogram(image2))
(vals2, bins2) = cum_hist(histogram(image3))
new_image = histogram_matching(image2, vals1, vals2)
plt.subplot(1, 3, 1)
plt.axis('off')
plt.title("cameraman.tif")
plt.imshow(image2, cmap = 'gray', aspect = "auto")
plt.subplot(1, 3, 2)
plt.axis('off')
plt.title("cell.tif")
plt.imshow(image3, cmap = 'gray', aspect = "auto")
plt.subplot(1, 3, 3)
plt.axis('off')
plt.title("Histogram Matching")
plt.imshow(new_image, cmap = 'gray', aspect = "auto")
plt.show()
"""

def midway(im1, c1, c2):
  temp = C(im1, c1)
  result = [[(cdf_inverse(val, c1) \
  + cdf_inverse(val, c2)) / 2.0 for val in row] for row in temp]
  return result

from skimage.color import rgb2gray
im1 = rgb2gray(plt.imread("Images/movie_flicker1.tif"))
im2 = rgb2gray(plt.imread("Images/movie_flicker2.tif"))
print(np.sum(im1))
print(im1.shape)
(vals1, bins1) = cum_hist(histogram(im1))
(vals2, bins2) = cum_hist(histogram(im2))
print("processing image1")
new_image1 = midway(im1, vals1, vals2)
print("processing image2")
new_image2 = midway(im2, vals2, vals1)
print("done prccessing image2")
print("np sum")
print(np.sum(im1 + new_image1))
plt.subplot(2, 2, 1)
plt.axis('off')
plt.title("movie_flicker1.tif")
plt.imshow(im1, cmap = 'gray', aspect = "auto")
plt.subplot(2, 2, 2)
plt.axis('off')
plt.title("movie_flicker2.tif")
plt.imshow(im2, cmap = 'gray', aspect = "auto")
plt.subplot(2, 2, 3)
plt.axis('off')
plt.title("movie_flicker1 midway specification")
plt.imshow(new_image1, cmap = 'gray', aspect = "auto")
plt.subplot(2, 2, 4)
plt.axis('off')
plt.title("movie_flicker 2 midway specification")
plt.imshow(new_image2, cmap = 'gray', aspect = "auto")
plt.show()





