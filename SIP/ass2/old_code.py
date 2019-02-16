import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import histogram
from matplotlib.pyplot import imread

def cum_hist(histogram):
  """Takes a histogram of length 256 with integer values as input
  and returns the cumulative histogram represented as an array
  from 0 to 255 with elements normalized from 0 to 1"""
  cum_his = np.cumsum(histogram)
  total = sum(histogram)
  normalized = cum_his / total 
  return np.round(normalized, 8)


def fix_bins(histogram_as_pair):
  (counts, bins) = histogram_as_pair
  result = [0 for i in range(256)]
  for i in range(min(bins), max(bins) + 1):
    result[i] = counts[i - min(bins)]
  return result
    

im1= imread("Images/pout.tif")
im2= rgb2gray(imread("Images/movie_flicker1.tif"))

from skimage.exposure import histogram
h1 = histogram(im1)
h2 = histogram(im2)
fix = fix_bins(h1)
result = cum_hist(fix)
