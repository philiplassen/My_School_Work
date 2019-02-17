#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:32:47 2019

@author: luise
"""


"""
###############################################################################
Question 3: Image filtering and enhancement
  3. mean and median filtering
  4. 
###############################################################################
"""


import matplotlib.pyplot as plt
import numpy as np
import time
import random
from scipy.stats import norm
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2hsv, hsv2rgb





import os

script_path = os.getcwd().split('\\')[-1]
image_path = script_path +'/images/'


img = imread(image_path + 'cameraman.tif')





def salt_and_pepper(img, p):
  """
  Put salt-and-pepper noise on an image: randomly selecting p/2 percent of pixel
  and putting them to black or white
  INPUT:
    - img: gray scaled image
    - p: persentage of the noise, 0 <= p <= 1
  OUTPUT:
    - img_noise: image with salt and pepper noise
  """
  
  img_noise = img
  #compute the number of pixels in the image
  size = img.shape
  pixel = size[0] * size[1]
  #choose random pixel number to be black(pepper)=0 or white(salt)=255
  number = int(np.round(pixel*p/2))
  noise = random.sample(range(1, pixel+1), number*2)
  pepper = noise[:number]
  salt = noise[number:]
  
  #overwrite image pixel in bw
  for i in range(number):
    x_white = salt[i] % size[0]
    y_white = int((salt[i] - x_white) / size[0])
    img_noise[x_white, y_white] = 255
    x_black = pepper[i] % size[0]
    y_black = int((pepper[i] - x_black) / size[0])
    img_noise[x_black, y_black] = 0

  return(img_noise)


def gaussian(img, sigma):
  """
  Put gaussian noise on an image
  INPUT:
    - img: gray scaled image
    - sigma: 
  OUTPUT:
    - img_noise: image with salt and pepper noise
  """
  
  size = img.shape
  #compute the gaussian nois and add it to the image
  noise = np.random.normal(0, sigma, size)
  img_noise = img + noise
  img_noise = np.round((img_noise/np.max(img_noise))*255)

  return(img_noise)



def median_filter_wrong(img, N):
  """
  Filters the image with an medean filter. The borders are set to zero. The
  output pixel is set to the median pixel value in the kernel
  INPUT:
    - img: gray scaled image
    - N: kernel size of the filter (needs to be odd)
  OUTPUT:
    - img_filter: filtered image
  """
  #check the kernel size
  if N%2 == 0:
    print('ERROR: kernel size N has to be odd')
    return
  
  size = img.shape
  #pad image
  border = int((N-1)/2)
  img_filter = np.pad(img, pad_width = border, mode='constant', constant_values=0)
  
  #running the kernel over the image
  for x in range(size[0]):
    for y in range(size[1]):
      kernel = img_filter[x:x+N, y:y+N]
      img_filter[int(x+np.floor(N/2)), int(y+np.floor(N/2))] = np.median(kernel)
  
  return(img_filter[border:size[0]+border-1, border:size[1]+border-1])

def median_filter(img, N):
  """
  Filters the image with an medean filter. The borders are set to zero. The
  output pixel is set to the median pixel value in the kernel
  INPUT:
    - img: gray scaled image
    - N: kernel size of the filter (needs to be odd)
  OUTPUT:
    - img_filter: filtered image
  """
  #check the kernel size
  if N%2 == 0:
    print('ERROR: kernel size N has to be odd')
    exit
  
  size = img.shape
  #pad image
  border = int((N-1)/2)
  img_filter = np.zeros(size)
  img_pad = np.pad(img, pad_width = border, mode='constant', constant_values=0)
  
  #running the kernel over the image
  for x in range(size[0]):
    for y in range(size[1]):
      kernel = img_pad[x:x+N, y:y+N]
      img_filter[x, y] = np.median(kernel)
  
  return(img_filter)



def mean_filter_wrong(img, N):
  """
  Filters the image with a mean filter. The borders are set to zero. The
  output pixel is set to the mean pixel value in the kernel
  INPUT:
    - img: gray scaled image
    - N: kernel size of the filter (needs to be odd)
  OUTPUT:
    - img_filter: filtered image
  """
  #check the kernel size
  if N%2 == 0:
    print('ERROR: kernel size N has to be odd')
    exit
  
  size = img.shape
  #pad image
  border = int((N-1)/2)
  img_filter = np.pad(img, pad_width = border, mode='constant', constant_values=0)
  
  #running the kernel over the image
  for x in range(size[0]):
    for y in range(size[1]):
      kernel = img_filter[x:x+N, y:y+N]
      img_filter[int(x+np.floor(N/2)), int(y+np.floor(N/2))] = int(round(np.mean(kernel)))
  
  return(img_filter[border:size[0]+border-1, border:size[1]+border-1])

def mean_filter(img, N):
  """
  Filters the image with a mean filter. The borders are set to zero. The
  output pixel is set to the mean pixel value in the kernel
  INPUT:
    - img: gray scaled image
    - N: kernel size of the filter (needs to be odd)
  OUTPUT:
    - img_filter: filtered image
  """
  #check the kernel size
  if N%2 == 0:
    print('ERROR: kernel size N has to be odd')
    exit
  
  size = img.shape
  #pad image
  border = int((N-1)/2)
  img_filter = np.zeros(size)
  img_pad = np.pad(img, pad_width = border, mode='constant', constant_values=0)
  
  #running the kernel over the image
  for x in range(size[0]):
    for y in range(size[1]):
      kernel = img_pad[x:x+N, y:y+N]
      img_filter[x, y] = int(round(np.mean(kernel)))
  
  return(img_filter)



def time_measure(N_list, img, number):
  """
  measures the time of median_filter and mean_filter number times
  INPUT:
    - N_list: different kernel sizes, which need to be tested
    - img: test image
    - number: how often the time was measured
  OUTPUT:
    - time_list_median: array of all different times [N, number] for median
    - time_list_mean: array of all different times [N, number] for mean
  """
  time_list_median = np.zeros((len(N_list), number))
  time_list_mean = np.zeros((len(N_list), number))

  for n in range(len(N_list)):
    for i in range(number):
      time1 = time.time()
      median_filter(img, N_list[n])
      time2 = time.time()
      time_list_median[n, i] = time2 - time1
      
      time1 = time.time()
      mean_filter(img, N_list[n])
      time2 = time.time()
      time_list_mean[n, i] = time2 - time1
    print('kernel size %i finished' %N_list[n])
      
  return(time_list_median, time_list_mean)




def gaussian_filter(img, N, sigma):
  """
  Filters the image with a gaussian filter. The borders are padded with zeros. 
  The output pixel is computed by the gaussian distributed input pixel of the j
  lernel. The kernel was linear seperated.
  INPUT:
    - img: gray scaled image
    - N: kernel size of the filter (needs to be odd)
    - sigma: standart derivation
  OUTPUT:
    - img_filter: filtered image
  """
  #check the kernel size
  if N%2 == 0:
    print('ERROR: kernel size N has to be odd')
    exit
  
  size = img.shape
  img_filter = np.zeros(size)
  #pad image
  border = int((N-1)/2)
  img_pad = np.pad(img, pad_width = border, mode='constant', constant_values=0)
  #computing the gaussian kernel
  kernel = np.arange(-N//2 +1, N//2 + 1)
  kernel = np.exp(-(kernel**2) / (2. * sigma**2))
  summe = sum(kernel)
  kernel = kernel/summe

  #linear fitting along the x-axis
  for x in range(size[0]):
    for y in range(size[1]):
      kernel_img = img_pad[x:x+N, y + N//2]
      img_filter[x, y] = sum(kernel*kernel_img)
  #linear fitting along the y-axis
  for y in range(size[1]):
    for x in range(size[0]):
      kernel_img = img_pad[x + N//2, y:y+N]
      img_filter[x, y] = sum(kernel*kernel_img)
  
  return(img_filter)







