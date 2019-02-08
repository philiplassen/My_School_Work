#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:27:59 2019

@author: luise
"""

"""
###############################################################################
Question 2:
  Select one of the greyscale images from the “Test Images” folder on the 
  course homepage. Implement a program to perform the bit-slicing technique as 
  described in the lecture slides, and extract/display the resulting bit-plane 
  images (see illustration in lecture slides) as separate images. Include your 
  Python code in your report, along with a figure showing the 8 bit-planes 
  (hint: use the subplot function).
###############################################################################
"""

from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte



def bit_slicing1(path, save_path):
  plt.figure()
  plt.subplot(3,3,1)
  I = img_as_ubyte(imread(path, as_gray = True))
  plt.imshow(I, cmap = 'gray')
  plt.axis('off')
  
  for b in range(0, 8):
    plt.subplot(3,3,b+2)
    J = np.bitwise_and(I, 2**b)
    plt.imshow(J, cmap = 'gray')
    plt.axis('off')
    
  plt.savefig(save_path + 'Q2_bit_slicing.png')
  
  

