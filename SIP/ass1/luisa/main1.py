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

import time
import matplotlib.pyplot as plt
import skimage.io as ski
from skimage.io import imread
from pylab import ginput
import numpy as np
from skimage.color import rgba2rgb
import numpy.random as npr
from skimage import img_as_ubyte
from skimage.color import rgb2hsv
from skimage import img_as_uint
from skimage.transform import resize
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
    
#motion(arr_path)


def resize_exp(path):

  I = imread(path)
  I = rgba2rgb(I)
  height = I.shape[0]
  length = I.shape[1]
  plt.subplot(2, 2, 1)
  plt.imshow(I)
  plt.axis('off')
  plt.title('orginal: (%i, %i)' %(height, length))
  
  I1 = resize(I, (np.floor(0.5*height), np.floor(0.5*length), 3))
  print(I.shape)
  print(I1.shape)
  print(type(I))
  print(type(I1))
  plt.subplot(2, 2, 2)
  plt.imshow(I1)
  plt.axis('off')
  plt.title('resize: 0.5')
  I2 = resize(I, (np.floor(0.2*height), np.floor(0.2*length), 3))
  plt.subplot(2, 2, 3)
  plt.imshow(I2)
  plt.axis('off')
  plt.title('resize: 0.2')

  I3 = resize(I, (np.floor(0.1*height), np.floor(0.1*length), 3))
  plt.subplot(2, 2, 4)
  plt.imshow(I3)
  plt.axis('off')
  plt.title('resize: 0.1')



def enlarge(img):

  height = img.shape[0]
  length = img.shape[1]
  factor = 5
  plt.subplot(2, 4, 1)#, sharex=True, sharey=True)
  plt.imshow(img)
  plt.axis('off')
  plt.title('orginal', fontsize = 12)

  location = [2, 3, 4, 6, 7, 8]

  for inter in range(6):
    time1 = time.time()*1000
    I1 = resize(img, (height*factor, length*factor, 3), order = inter)
    time2 = time.time()*1000
    diff = time2 - time1
    plt.subplot(2, 4, location[inter])
    plt.imshow(I1)
    plt.axis('off')
    plt.title('order: %i\n time: %.2fs' %(inter, diff), fontsize = 10)
  plt.show()

I = imread('Images/tree.png')
I = rgba2rgb(I)
img = resize(I, (np.floor(0.2*I.shape[0]), np.floor(0.2*I.shape[1]), 3))

resize_exp('Images/tree.png')
enlarge(img)
