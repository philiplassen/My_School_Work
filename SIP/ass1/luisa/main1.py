#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:12:17 2019

@author: luise
"""

"""
###############################################################################
          ASSIGNMENT 1
###############################################################################


#import other files
import Q1
import Q2
import Q3
"""
#import packagespath = image_path + 'AT3_1m4_01.png'
import os #path


import matplotlib.pyplot as plt
import skimage.io as ski
from skimage.io import imread
from pylab import ginput
import numpy as np
import numpy.random as npr
from skimage import img_as_ubyte
from skimage.color import rgb2hsv
from skimage import img_as_uint

script_path = os.getcwd().split('\\')[-1]
save_path = script_path +'/images/'
image_path = script_path +'/images/Test_Images_export/'



## Exercise 1 -----------------------------------------------------------------

#Q1.black_pixel(".")


## Exercise 2 -----------------------------------------------------------------

#Q2.bit_slicing1(image_path + 'bigben.png', save_path)


## Exercise 3 -----------------------------------------------------------------

#Q3.HSV("../Images/" + 'peppers.png')


## Exercise 6 -----------------------------------------------------------------
"""
Q6.resize_exp(image_path + 'brasil.JPG', save_path)
"""

## Exercise 9 -----------------------------------------------------------------
arr_path = ["Images/AT3_1m4_0" + str(i) + ".tif" for i in range(1, 10)]
arr_path += ['Images/AT3_1m4_10.tif']
def motion(arr_path):
  plt.figure()
  number = len(arr_path)
  for i in range(number-1):
    img1 = imread(arr_path[i])
    img2 = imread(arr_path[i+1])
    img = img_as_uint(np.abs(img1.astype('int16') - img2))
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
  plt.show()
    
motion(arr_path)


