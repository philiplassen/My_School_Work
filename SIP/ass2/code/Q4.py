#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:26:35 2019

@author: luise
"""


"""
###############################################################################
Question 2: Bonus question
  3. Cummulative histogram
  4. 
###############################################################################
"""
#

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
import numpy as np
import time



def gaussian_1d(x, sigma):
  return(np.exp(-x**2/(2*sigma**2)))

def gaussian_2d(x, y, sigma):
  return(np.exp(-(x**2 + y**2)/(2*sigma**2)))

def w(I, x, y, i, j, sigma, tau):
  return(gaussian_2d(i, j, sigma) * gaussian_1d((I[x+i, y+j] - I[x, y]), tau))



def bilateral_filter(img, N, sigma, tau):
  """
  Filters the image with a bilateral filter. The outer pixel are not filtered. 
  So the output image is smaller than the input image.
  The output pixel is computed by 
  INPUT:
    - img: gray scaled image
    - N: kernel size of the filter (needs to be odd)
    - sigma, tau: standart derivation
  OUTPUT:
    - img_filter: filtered image
  """  
  size = img.shape
  img_filter = np.zeros((size[0], size[1]))
  z=0; n=0
  
  #running the kernel over the image
  for x in range(N//2, size[0]-N//2):
    for y in range(N//2, size[1]-N//2):
      #running pixel-wise over the kernel
      for j in range(-(N//2), int(np.ceil(N//2))):
        for i in range(-(N//2), int(np.ceil(N//2))):
          z += w(img, x, y, i, j, sigma, tau)*img[x+i, y+i]
          n += w(img, x, y, i, j, sigma, tau)
         
      img_filter[x, y] = z/n
      z=0; n=0
   
  return(img_filter[N//2:-(N//2+1), N//2:-(N//2+1)])



