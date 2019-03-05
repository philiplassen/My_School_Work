#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 08:46:44 2019

@author: luise
"""

"""
###############################################################################
All functions used in the 3rd Assignment:
  Fourier Transformation and Morphology
###############################################################################
"""

#import packages
import os #path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
#import skimage
import matplotlib
from skimage import io
from skimage import color
from matplotlib import cm
import time
import random

"""
### FOURIER TRANSFORMATION ----------------------------------------------------
1) Write the function scale, which implements convolution with an isotropic
Gaussian kernel, parametrized by its standard deviation σ - the scale.
Apply it to trui.png for a range of scales.
"""



## 1) Convolution with an isotropic Gaussian kernel

def scale(img, N, sigma):
  """
  Filters the image with a gaussian filter. Convolution was implemented with
  fourier transformation.
  INPUT:
    - img: gray scaled image
    - N: kernel size of the filter (needs to be odd)
    - sigma: standart derivation
  OUTPUT:
    - img_filter: filtered image
  """
  
  size = img.shape
  img_filter = np.copy(img)
  #computing the gaussian kernel
  kernel_axis = np.arange(-N//2 +1, N//2 +1)
  xx, yy = np.meshgrid(kernel_axis, kernel_axis)
  kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
  summe = np.sum(kernel)
  kernel = kernel/summe

  #convolution by convertion into frequency domain
  #fourier transformation
  img_filter_fft = fft2(img_filter)
  kernel_fft = fft2(kernel, size)
  #convolution in fourier domain = multiplication
  img_filter_fft = img_filter_fft*kernel_fft
  #transform it back
  img_filter = ifft2(img_filter_fft)
  
  return(np.absolute(img_filter))


def derive(img, x_ord, y_ord):                                                    #is the formula for the derivative right?!?!?!?!
  """
  Computes the derivative of an image by transfomring it into th eFourier domain
  INPUT:
    - img: image which needs to be derived
    _ x_ord, y_ord: order of the dirivative in the oriantation of each axis
  OUTPUT:
    - img_derive: derivative of the image
  
  """
  size = img.shape
  #fourier transformation
  img_fft = fft2(img)
  #computation derivative multiplicator
  x = np.arange(size[1])
  y = np.arange(size[0])
  xx, yy = np.meshgrid(x, y)
  kernel = (2*np.pi*1.j*xx)**x_ord/size[1] * (2*np.pi*1.j*yy)**y_ord/size[0]
  #derivative in fourier domain
  img_fft_derive = img_fft*kernel
  #transform it back
  img_derive = ifft2(img_fft_derive)
  
  return(np.absolute(img_derive))
















