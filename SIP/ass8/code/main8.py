#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:29:05 2019

@author: luise
"""

"""
###############################################################################
          ASSIGNMENT 8 - Group
###############################################################################
"""


#import other files
import functions as fct

#import packages
import os #path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import hough_line




script_path = os.getcwd().split('\\')[-1]
image_path = script_path +'/images/'



"""
EXERCISE 1.3:
  Implement a histogram based segmentation method that given a grayscale image 
  analyzes the histogram to automatically and a good threshold. Test the method
  on the images coins.png and overlapping_euros1.png.
"""

img_coin = io.imread(image_path + 'coins.png')
img_euro = io.imread(image_path + 'overlapping_euros1.png')

#compute the histograms with numpy
his_coin = np.histogram(img_coin, bins  = 256)
his_euro = np.histogram(img_euro, bins  = 256)
#plot the histograms
fig, ax = plt.subplots(2, 2, figsize = (7,5))
ax[0, 0].imshow(img_coin, cmap = 'gray')
ax[0, 0].axis('off')
ax[0, 0].set_title('coins.png')
ax[0, 1].imshow(img_euro, cmap = 'gray')
ax[0, 1].axis('off')
ax[0, 1].set_title('overlapping_euros1.png')
ax[1, 0].hist(img_coin.flatten(), bins = 256)
ax[1, 0].set_title('intensity histogram')
ax[1, 0].set_xlabel('intensity')
ax[1, 0].set_ylabel('counts')
ax[1, 1].hist(img_euro.flatten(), bins = 256)
ax[1, 1].set_title('intensity histogram')
ax[1, 1].set_xlabel('intensity')

fig.savefig(image_path + '1_histo.png', bbox = 'tight')

#compute the threshhold
thresh_coin, img_coin_thresh = fct.segmentation_histo(img_coin)
thresh_euro, img_euro_thresh = fct.segmentation_histo(img_euro)
#Visualization of the threshhold
fig, ax = plt.subplots(2, 2, figsize = (7,5))
ax[0, 0].imshow(img_coin, cmap = 'gray')
ax[0, 0].axis('off')
ax[0, 0].set_title('coins.png')
ax[0, 1].imshow(img_euro, cmap = 'gray')
ax[0, 1].axis('off')
ax[0, 1].set_title('overlapping_euros1.png')
ax[1, 0].imshow(img_coin_thresh, cmap = 'gray')
ax[1, 0].set_title('threshhold = %i' %thresh_coin)
ax[1, 0].axis('off')
ax[1, 1].imshow(img_euro_thresh, cmap = 'gray')
ax[1, 1].set_title('threshhold  = %i' %thresh_euro)
ax[1, 1].axis('off')

fig.savefig(image_path + '1_thresh.png', bbox = 'tight')




"""
EXERCISE 2.1:
  Implement straight line Hough Transform, describe how your implementation 
  works and possible issues your implementation can have (e.g. computational 
  complexity).
"""
img_cross = io.imread(image_path + 'cross.png')

cross_hough, angles_cross, dist_cross = fct.hough_trans_line(img_cross)
plt.imshow(cross_hough, cmap = 'gray')



"""
EXERCISE 2.2:
  Test your implementation on the image cross.png. Plot both the Hough transform
  of the image and the detected lines overlayed on cross.png. Compare the 
  output of your implementation to the output of the method hough_line in 
  scikit-image.
"""
x = np.arange(5, 95, 1)
#cross of our implementation
angle_cross, dis_cross = fct.hough_trans_line_peak(cross_hough, angles_cross, dist_cross, k = 40)
y0 = (dis_cross[0] - x*np.cos(angle_cross[0]))/np.sin(angle_cross[0])
y1 = (dis_cross[1] - x*np.cos(angle_cross[1]))/np.sin(angle_cross[1])

#referenz with cross
cross_hough_ref, angles_cross_ref, dist_cross_ref = hough_line(img_cross)
angle_cross_ref, dis_cross_ref = fct.hough_trans_line_peak(cross_hough_ref, angles_cross_ref, dist_cross_ref, k = 45)
y0_ref = (dis_cross_ref[0] - x*np.cos(angle_cross_ref[0]))/np.sin(angle_cross_ref[0])
y1_ref = (dis_cross_ref[1] - x*np.cos(angle_cross_ref[1]))/np.sin(angle_cross_ref[1])

fig, axes = plt.subplots(2, 2, figsize=(7, 8))
axes[0, 0].imshow(img_cross, cmap='gray')
axes[0, 0].set_title('own implementation')
axes[0, 0].plot(x, y0, color = 'r')
axes[0, 0].plot(x, y1, color = 'r')
axes[1, 0].imshow(np.log(1 + cross_hough), cmap='gray', extent=(np.rad2deg(angles_cross[-1]), np.rad2deg(angles_cross[0]), dist_cross[-1], dist_cross[0]))
axes[1, 0].set_title('Hough transform')
axes[1, 0].set_xlabel('Angle (degree)')
axes[1, 0].set_ylabel('Distance (pixel)')

axes[0, 1].imshow(img_cross, cmap='gray')
axes[0, 1].set_title('scikit-image')
axes[0, 1].plot(x, y0_ref, color = 'r')
axes[0, 1].plot(x, y1_ref, color = 'r')
axes[1, 1].imshow(np.log(1 + cross_hough_ref), cmap='gray', extent=(np.rad2deg(angles_cross_ref[-1]), np.rad2deg(angles_cross_ref[0]), dist_cross_ref[-1], dist_cross_ref[0]))
axes[1, 1].set_title('Hough transform')
axes[1, 1].set_xlabel('Angle (degree)')
axes[1, 1].set_ylabel('Distance (pixel)')

plt.tight_layout()
fig.savefig(image_path + '2_hough_trans.png', bbox = 'tight')



"""
EXERCISE 2.3:
  Use the  hough_circle method of scikit-image to make a segmentation method that can 
  segment the coins in  coins.png. Describe the method and visualize 
  the results. You will likely need to apply the edges in coins.png 
  hough_circle to instead of the actual image, e.g. run Canny edge 1detection 
  from scikit-image and feed the output to the circle Hough transform.
"""
coin_seg, hough_coin = fct.segmentation_hough(img_coin)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img_coin, cmap = 'gray')
ax[0].set_title('original image', fontsize = 15)
ax[0].axis('off')
ax[1].imshow(coin_seg, cmap = 'gray')
ax[1].set_title('segmentation', fontsize = 15)
ax[1].axis('off')
ax[2].imshow(hough_coin, cmap = 'gray')
ax[2].set_title('Hough transform', fontsize = 15)
ax[2].axis('off')

plt.tight_layout()
fig.savefig(image_path + '2_hough_circle.png', bbox = 'tight')



