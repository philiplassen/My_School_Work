#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:14:45 2019

@author: luise
"""


"""
###############################################################################
Question 1: Pixel-wise contrast enhancement
  1. Gamma transform of a gray scale image
  2. Gamma transform of a rgb image
  3. Gamma transform of a hsv image
###############################################################################
"""


from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
import numpy as np
import time



def gamma_transform(img, gamma):
  """
  Gamma transform of a gray scale image. It's taken care of the output pixel 
  range from 0 to 255
  INPUT:
    - img: gray scaled image
    - gamma: gamma value
  OUTPUT:
    - img_trans: gamma transformed image
  """
  
  img_trans = np.power(img, gamma)
  img_trans = np.round((img_trans/np.max(img_trans))*255)
  return(img_trans)
  
  
  
def gamma_transform_rgb(img, gamma):
  """
  Gamma transform of a rgb image. Transforming each coulor channel seperatly
  INPUT:
    - img: rgb image
    - gamma: gamma value
  OUTPUT:
    - img_trans: gamma transformed image
  """
  img_shape = img.shape
  img_trans = np.zeros(img_shape)
  for i in range(3):
    img_trans[:, :, i] = gamma_transform(img[:, :, i], gamma)
    #img_trans[:, :, i] = np.round((transformed_channel/np.max(transformed_channel))*255)
  return(img_trans)



def gamma_transform_hsv(img, gamma):
  """
  Gamma transform of a rgb image. Transforming to hsv and then each coulor 
  transforming each channel seperatly
  INPUT:
    - img: rgb image
    - gamma: gamma value
  OUTPUT:
    - img_trans: gamma transformed image
  """
  img_trans = rgb2hsv(img)
  img_trans[:, :, 2] = gamma_transform(img_trans[:, :, 2], gamma)
  img_trans = hsv2rgb(img_trans)
  return(img_trans)











