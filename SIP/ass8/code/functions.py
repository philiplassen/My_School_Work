#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:29:05 2019

@author: luise
"""

"""
###############################################################################
All functions used in the 8th Assignment
###############################################################################
"""


#import packages
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage.feature import peak_local_max
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks
from scipy.signal import argrelextrema
import cv2 as cv


def find_threshold(img, order = 10):
  """
  finds the optimal threshold.
  optimal: the nearest local minimum of the intensity histogram next to the
  mean intensity value of the image. The variable order defines how many
  neigbours are included to find the local minimum.
  """
  # mean intensity
  img_mean = np.mean(img)
  #local min
  histo = np.histogram(img, bins  = 256)
  min_loc = argrelextrema(histo[0], np.less, order=order )
  #local min nearest to mean
  min_val = np.argmin(np.absolute(min_loc - img_mean))
  #threshold value
  threshold = min_loc[0][min_val]
  return(threshold)



def segmentation_histo(img, order = 10): 
  """
  segmentation of a gray scale image
  """
  threshold = find_threshold(img, order)
  #print(threshold)
  #threshold the image
  img_thresh = cv.threshold(img,threshold,255,cv.THRESH_BINARY)
  return(threshold, img_thresh[1])
  
  
  
def hough_trans_line(img):
  """
  straight line Hough Transform. The input image has to be a edge detection
  (means a binary image).
  """
  #define variables
  width, height = img.shape
  thetas = np.deg2rad(np.arange(-90., 90.)) #angle
  diag = np.ceil(np.sqrt(width**2 + height**2)) #max distance is diagonal: pythagoras
  rhos = np.arange(-diag, diag+1) #distance
  output = np.zeros((len(rhos), len(thetas)))

  #run over all non zero values -> edges
  edges_y, edges_x = np.nonzero(img)
  for i in range(len(edges_y)):
    x = edges_x[i]
    y = edges_y[i]
    
    #run over all thetas cause of the formula: rho = x*cos(theta) + y*sin(theta)
    for t in range(len(thetas)):
      theta = thetas[t]
      rho = x*np.cos(theta) + y*np.sin(theta)
      #print('t', theta, 'r', rho)
      #rho index
      r = np.argmin(np.absolute(rhos-rho))
      #hough transform:output[r, t] will be added by 1
      output[r, t] += 1

  return output, thetas, rhos #thus we have the same output as scikit-image



def hough_trans_line_peak(hough, angles, dis, k):
  """ 
  Finds the maximum peaks in the hough domain
  """
  lmc = peak_local_max(hough, min_distance=k)
  print(lmc)
  angele_ind = np.array([angles[lmc[0][1]], angles[lmc[1][1]]])
  dis_ind = np.array([dis[lmc[0][0]], dis[lmc[1][0]]])

  return angele_ind, dis_ind


def make_circle(size, x_cord, y_cord, rad):
  """
  creates a binary image, with white circles
  """
  mask = np.zeros(size)
  Y = size[0]
  X = size[1]
  
  for i in range(Y):
      for j in range(X):
        for c in range(len(x_cord)):
          r_ref = np.linalg.norm([i-y_cord[c], j-x_cord[c]])
          if r_ref < rad[c]:
              mask[i,j] = 1
  
  return mask



def segmentation_hough(img, sigma = 3.5, radii_lim = [20, 50], num_peaks = 12):
  """
  segmentation of a grayscale image with circula Hough Transform
  """
  #edge detection
  edges = feature.canny(img, sigma=sigma)
  #circular Hough Transform with different radii
  radii = np.arange(radii_lim[0], radii_lim[1], 2)
  hough = hough_circle(edges, radii)
  #plt.imshow(np.max(hough, axis=0), cmap = 'gray')
  hough_par, x_par, y_par, rad_par = hough_circle_peaks(hough, radii, 
                                                        total_num_peaks=num_peaks)
  #draw segmentation image
  img_seg = make_circle(img.shape, x_par, y_par, rad_par)
  return img_seg, np.max(hough, axis=0)







