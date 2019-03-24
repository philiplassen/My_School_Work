#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:17:07 2019

@author: luise
"""

"""
###############################################################################
All functions used in the 7th Assignment
###############################################################################
"""

#import packages
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from scipy import signal
from scipy.ndimage import rotate
from skimage.feature import peak_local_max



def norm_hist(histo):
  """
  returns the normalized histogram: the sum of all categories is one
  """
  summe = sum(histo[0])
  return( histo[0]/summe, histo[1] )


def fill_up_arr(arr, n):
  """
  creates an array with n elements where all the missing elements of arr[0] are
  set to zero
  """
  compare_arr = range(0, n+1)
  output = []
  dummy = 0
  for c in compare_arr:
    if c == arr[1][dummy]:
      output.append(arr[0][dummy])
      dummy +=1
    else:
      output.append(0)
    
    if dummy == len(arr[1]):
      if len(arr[1]) != n+1:
        diff = n - len(output) +1
        output = np.pad(output, (0, diff), 'constant')
      return(np.array(output))

def chi2_distance(histo1, histo2, num, fill = True):
  """
  returns the Chi^2 distance of two histograms hist1, histo2. If both bins in 
  the two histograms are zero, the distance is set to zero.
  """
  if fill:
    h = fill_up_arr(histo1, num)
    g = fill_up_arr(histo2, num)
  else:
    h = histo1[0]
    g = histo2[0]
  output = []
  for i in range(num+1):
    if g[i]+h[i] != 0:
      output.append((h[i]-g[i])**2/(h[i]+g[i]))
    else:
      output.append(0)
  return(sum(output))



def Gaussian(x, y, n, m, sigma):
  """
  image of a discrete Gaussian distribution: only for the 0th, 1st and 2dn 
  derivative! n in x-direction and m in y direction
    n and m = 0: Gaussian
    n or m = 1: first derivative
    n and m = 1: second derivative
    n or m = 2: second derivative
  """
  x_axis = np.arange(-x//2 +1, x//2 +1)
  y_axis = np.arange(-y//2 +1, y//2 +1)
  xx, yy = np.meshgrid(x_axis, y_axis)
  
  #normal Gaussian
  gauss = 1/(2.*np.pi*sigma**2)*np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
  if n==0 and m==0:
    gauss = gauss
  #first derivative of the Gaussian
  elif n==1 and m==0:
    gauss = -xx/sigma**2 * gauss
  elif n==0 and m==1:
    gauss = -yy/sigma**2 * gauss
  #second order derivative
  elif n==2 and m==0:
    gauss = 1/sigma**2 * (xx**2/sigma**2 -1) * gauss
  elif n==0 and m==2:
    gauss = 1/sigma**2 * (yy**2/sigma**2 -1) * gauss
  elif n==1 and m==1:
    gauss = xx/sigma**2 * yy/sigma**2 * gauss
  else:      
    gauss = None
    
  return(gauss)


def Lxy(img, x, y, n, m, sigma):
  """
  filter response image of image img: the responses to Gaussian G(x, y; sigma)
  (means n, m = 0) or the derivative of an Gaussian filter of order n+m
  """
  kernel = Gaussian(x, y, n, m, sigma)
  #response_img = signal.convolve2d(img, kernel, mode='same')
  response_img = ifft2(fft2(img) * fft2(kernel, img.shape))
  
  return np.real(response_img)


def Lv(img, x, y, sigma, theta):
  """
  filter response image of image img: the responses to Gaussian derivative 
  filter of first order in the orientation theta (in rad)
  """
  response_img = np.cos(theta) * Lxy(img, x, y, 1, 0, sigma) 
  + np.sin(theta) * Lxy(img, x, y, 0, 1, sigma)
  
  return response_img


def Lvv(img, x, y, sigma, theta):
  """
  filter response image of image img: the responses to Gaussian derivative 
  filter of second order in the orientation theta (in rad)
  """
  response_img = np.cos(theta)**2 * Lxy(img, x, y, 2, 0, sigma) 
  + np.sin(theta)**2 * Lxy(img, x, y, 0, 2, sigma) 
  + 2 * np.cos(theta) * np.sin(theta) * Lxy(img, x, y, 1, 1, sigma)
  
  return response_img


def max_Lv(img, x, y, sigma, theta_arr):
  """
  computes the maximal image of all oriented Lv(theta) derivative filters
  """
  Lv_images = np.zeros((img.shape[0], img.shape[1], len(theta_arr)))
  for t in range(len(theta_arr)):
    Lv_images[:, :, t] = Lv(img, x, y, sigma, theta_arr[t])
  Lv_max = np.max(Lv_images, axis = 2)
  return Lv_max


def max_Lvv(img, x, y, sigma, theta_arr):
  """
  computes the maximal image of all oriented Lvv(theta) derivative filters
  """
  Lvv_images = np.zeros((img.shape[0], img.shape[1], len(theta_arr)))
  for t in range(len(theta_arr)):
    Lvv_images[:, :, t] = Lvv(img, x, y, sigma, theta_arr[t])
  Lvv_max = np.max(Lvv_images, axis = 2)
  return Lvv_max


def MR8_response(img, sigma, sigma_der, theta):
  """
  computes all 8 response images: Gaussian, Laplacian and the orientated first
  and second order derivatives of the Gaussian. The output is of all
  images (also the original image) stuck together in an array.
  """
  N = 6* sigma +1
  sigma_test_der = [1, 2, 4]
  ax_der = [6*s +1 for s in sigma_der]
  
  img_response = np.zeros((img.shape[0], img.shape[1], 9))
  img_response[:, :, 0] = img
  #Gaussian
  img_response[:, :, 1] = Lxy(img, N, N, 0, 0, sigma)
  #Laplacian
  img_response[:, :, 2] = Lxy(img, N, N, 2, 0, sigma) + Lxy(img, N, N, 0, 2, sigma)
  #orientated derivatives
  for i in range(len(sigma_test_der)):
    img_response[:, :, 3+i] = max_Lv(img, ax_der[i], ax_der[i], sigma_test_der[i], theta)
  for i in range(len(sigma_test_der)):
    img_response[:, :, 6+i] = max_Lvv(img, ax_der[i], ax_der[i], sigma_test_der[i], theta)
    
  return img_response








