#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:14:44 2019

@author: luise
"""

"""
###############################################################################
          ASSIGNMENT 2
###############################################################################
"""

#import other files
import Q1
#import Q2
import Q3
import Q4

#import packagespath = image_path + 'AT3_1m4_01.png'
import os #path
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt


script_path = os.getcwd().split('\\')[-1]
image_path = script_path +'/images/'




## Question1: Pixel-wise contrast enhancement ---------------------------------

### 1.1 Gamma correction gray scale
img_cameraman = imread(image_path + 'cameraman.tif')
gamma_list = [3/2, 1, 1/2, 1/3, 1/4, 1/5]

plt.figure()
for i in range(len(gamma_list)):
  img_cameraman_trans = Q1.gamma_transform(img_cameraman, gamma_list[i])
  
  plt.subplot(2, 3, i+1)
  plt.imshow(img_cameraman_trans, cmap = 'gray')
  plt.title('$\gamma$ = %.2f' %gamma_list[i])
  if gamma_list[i] == 1:
    plt.title('$\gamma$ = %.2f (orginal)' %gamma_list[i])
  plt.axis('off')

plt.savefig(image_path + '1_gammaTransform.png')


### 1.2 Gamma correction rgb
img_autumn = imread(image_path + 'autumn.tif')

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_autumn)
plt.axis('off')
plt.title('orginal (rgb)')
plt.subplot(1, 2, 2)
img_autumn_trans = Q1.gamma_transform_rgb(img_autumn, .7)
#plt.imshow(img_autumn_trans)
plt.imshow((img_autumn_trans).astype(np.uint8))
plt.axis('off')
plt.title('rgb transformed\n $\gamma$ = 0.70')

plt.savefig(image_path + '1_gammaTransform_rgb.png')


### 1.3 Gamma correction hsv
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_autumn)
plt.axis('off')
plt.title('orginal rgb')
#plt.subplot(1, 3, 3)
#img_autumn_trans_rgb = Q1.gamma_transform_rgb(img_autumn, .7)
#plt.imshow(img_autumn_trans_rgb.astype(np.uint8))
#plt.axis('off')
#plt.title('rgb transformed\n $\gamma$ = 0.70')
plt.subplot(1, 2, 2)
img_autumn_trans_hsv = Q1.gamma_transform_hsv(img_autumn, .7)
plt.imshow(img_autumn_trans_hsv.astype(np.uint8))
plt.axis('off')
plt.title('hsv transformed\n $\gamma$ = 0.70')

plt.savefig(image_path + '1_gammaTransform_hsv.png')











## Question3: Image filtering and enhancement ---------------------------------

### 3.3 Mean and Median filtering
#compare mean and median filter
img_eight = imread(image_path + 'eight.tif')
plt.figure(figsize=(9,7))

plt.subplot(3, 3, 1)
plt.imshow(img_eight, vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('orginal')
plt.subplot(3, 3, 4)
plt.imshow(Q3.median_filter(img_eight, 3), vmin=0, vmax=255, cmap = 'gray')
plt.xlabel('median/n filter')
plt.axis('off')
plt.subplot(3, 3, 7)
plt.imshow(Q3.mean_filter(img_eight, 3), vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.xlabel('mean/n filter')

img_eight_sp = Q3.salt_and_pepper(img_eight, 0.05)
plt.subplot(3, 3, 2)
plt.imshow(img_eight_sp, vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('salt and pepper\n p=0.05')
plt.subplot(3, 3, 5)
plt.imshow(Q3.median_filter(img_eight_sp, 3), vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.subplot(3, 3, 8)
plt.imshow(Q3.mean_filter(img_eight_sp, 3), vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')

img_eight = imread(image_path + 'eight.tif')
img_eight_g = Q3.gaussian(img_eight, 10)
plt.subplot(3, 3, 3)
plt.imshow(img_eight_g, vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('gaussian\n $\sigma$=10')
plt.subplot(3, 3, 6)
plt.imshow(Q3.median_filter(img_eight_g, 3), vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.subplot(3, 3, 9)
plt.imshow(Q3.mean_filter(img_eight_g, 3), vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')

plt.savefig(image_path + '3_mean_med_filter.png')


#different kernel sizes
plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.imshow(img_eight_sp, vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('no filter')
plt.subplot(1, 4, 2)
plt.imshow(Q3.median_filter(img_eight_g, 3), vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('kernel: N = 3')
plt.subplot(1, 4, 3)
plt.imshow(Q3.median_filter(img_eight_g, 7), vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('kernel: N = 7')
plt.subplot(1, 4, 4)
plt.imshow(Q3.median_filter(img_eight_g, 11), vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('kernel: N = 11')

plt.savefig(image_path + '3_med_kernel.png')

"""
#compare the runtime with different kernel sizes
kernel_list = (np.arange(1, 13)*2)+1
number = 100

median_time, mean_time = Q3.time_measure(kernel_list, img_eight_sp, number)
median_time_mean = np.mean(median_time, axis = 1)
mean_time_mean = np.mean(mean_time, axis = 1)

plt.plot(kernel_list, median_time_mean, label = 'median filter')
plt.plot(kernel_list, mean_time_mean, color = 'red', label = 'mean filter')
plt.xlabel('kernel size')
plt.ylabel('runtime [s]')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(image_path + '3_runtime1.png', dpi=300, format='png', bbox_inches='tight')
"""

#gaussian filter: kernel size
std = 5
kernel_list = [1, 3, 5, 7, 9, 15, 19, 25, 35]

plt.figure(figsize=(6, 6))
for n in range(len(kernel_list)):
  if n == 0:
    plt.subplot(3, 3, n+1)
    plt.imshow(img_eight_sp, vmin=0, vmax=255, cmap = 'gray')
    plt.axis('off')
    plt.title('no filter')
  else:
    plt.subplot(3, 3, n+1)
    plt.imshow(Q3.gaussian_filter(img_eight_sp, kernel_list[n], std), vmin=0, vmax=255, cmap = 'gray')
    plt.axis('off')
    plt.title('kernel: N = %i' %kernel_list[n])

plt.savefig(image_path + '3_gauss_kernel.png')


#gaussian filter: std
std_list = [1, 3, 5, 7, 9, 11, 15, 17, 19]

plt.figure(figsize=(6, 6))
for std in range(len(std_list)):
  kernel = int(np.ceil(std_list[std]*5))
  plt.subplot(3, 3, std+1)
  plt.imshow(Q3.gaussian_filter(img_eight_sp, kernel, std_list[std]), vmin=0, vmax=255, cmap = 'gray')
  plt.axis('off')
  plt.title('std: $\sigma$ = %i\n kernel: N = %i' %(std_list[std], kernel))

plt.savefig(image_path + '3_gauss_std.png')





## Question4: Bilateral filtering ---------------------------------------------
###4.2 bilateral filter
sigma = 10
tau = 5

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(img_eight_sp, vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('no filter\n')
plt.subplot(1, 3, 2)
N=5
img_eight_filtered = Q4.bilateral_filter(img_eight_sp, N, sigma, tau)
plt.imshow(img_eight_filtered, vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('bilateral filter\n N=%i, $\sigma$=%.1f, $\\tau$=%.1f' %(N, sigma, tau))
plt.subplot(1, 3, 3)
N=10
img_eight_filtered = Q4.bilateral_filter(img_eight_sp, N, sigma, tau)
plt.imshow(img_eight_filtered, vmin=0, vmax=255, cmap = 'gray')
plt.axis('off')
plt.title('bilateral filter\n N=%i, $\sigma$=%.1f, $\\tau$=%.1f' %(N, sigma, tau))

plt.savefig(image_path + '4_bilateral.png')










