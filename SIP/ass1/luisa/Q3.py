#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:26:16 2019

@author: luise
"""


"""
###############################################################################
Question 3:
  Pick one of your own photos and import it into Python (use ‘png’, ‘tif ’ or 
  ‘jpg’ format) and check that it is in RGB format. Using the 
  skimage.color.rgb2hsv function, write a program to display the individual 
  hue, saturation and value channels of your (or any other) RGB colour image. 
  You may wish to refer to the reading material (Gonzales & Woods) Figure 6.16. 
  Include the Python code and resultant images in your report.
###############################################################################
"""

from skimage.io import imread
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt


"""
def is_not_RGB(img):
  
  if len(img.shape) == 3:
    return False
  else:
    return True
"""


def HSV(path):
  
  plt.figure()
  I = imread(path)
  
  if len(I.shape) != 3:
    print('ERROR: image not in RGB')
    exit
  
  plt.subplot(2,2,1)
  plt.imshow(I)
  plt.axis('off')
  plt.title('original')
  
  I_hsv = rgb2hsv(I)
  
  plt.subplot(2,2,2)
  plt.imshow(I_hsv[:, :, 0], cmap = 'hsv')
  plt.axis('off')
  plt.title('hue channel')
  
  plt.subplot(2,2,3)
  plt.imshow(I_hsv[:, :, 1], cmap = 'gray')
  plt.axis('off')
  plt.title('saturation channel')
  
  plt.subplot(2,2,4)
  plt.imshow(I_hsv[:, :, 2], cmap = 'gray')
  plt.axis('off')
  plt.title('value channel')
  plt.show()
  #plt.savefig(save_path + 'Q3_hsv.png')

