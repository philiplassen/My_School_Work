#!/usr/bin/env python3


# Importing packages
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
import numpy as np
import matplotlib.pyplot as plt
import skimage
import matplotlib
from skimage import io
from skimage import color
from matplotlib import cm


import sys
DEBUG = False
A = False
B = False
C = False
for arg in sys.argv:
  if arg == '-v':
    DEBUG = True
  if arg == '-a':
    A = True
  if arg == '-b':
    B = True
  if arg == '-c':
    C = True
  
#Verbose for Debugging
def log(message):
  if DEBUG:
    print(message)



matplotlib.rcParams['figure.figsize'] = (4,4)



def trans(f):
  f_fft = fft2(f)
  f_fft_shift = fftshift(f_fft)
  f_fft_abs = np.absolute(f_fft_shift)
  power = f_fft_abs ** 2
  f_logtrans = 100*np.log(1 + power)
  return f_logtrans
 


if A:
  f = np.array(color.rgb2gray(io.imread('trui.png').astype(float)))
  f_logtrans = trans(f)
  fig, ax = plt.subplots(1,2)
  ax[0].axis("off")
  ax[0].set_title("Original Image")
  ax[0].imshow(f, cmap=cm.Greys_r)
  ax[1].axis("off")
  ax[1].set_title("Log Transformed Spectrum")
  ax[1].imshow(f_logtrans, cmap=cm.Greys_r)
  plt.show()



def convolve_fft(kernel, image_path):
  f = np.array(color.rgb2gray(io.imread('trui.png').astype(float)))
  (rows, columns) = f.shape
  (height, width) = kernel.shape
  f_fft = fft2(f)
  k_fft = fft2(kernel, shape = f.shape)
  result = ifft2(f_fft * k_fft)
  return result[height - 1: rows - height + 1, width - 1:columns - width + 1]

  
def convolve_brute(kernel, image_path):
  f = np.array(color.rgb2gray(io.imread('trui.png').astype(float)))
  k = np.rot90(kernel, 2)
  (rows, columns) = f.shape
  (height, width) = k.shape
  clipped  = f[height - 1: rows - height + 1, width - 1:columns - width + 1]
  filtered = np.zeros(clipped.shape)
  for r in range(height - 1, rows - height + 1):
    for c in range(width - 1, columns - width + 1):
      (r_offset, c_offset) = (height // 2, width // 2)
      result = np.sum(k * f[(r - r_offset):(r + r_offset + 1), (c - c_offset):(c + c_offset + 1)])
      filtered[r - (height - 1), c - (width - 1)] = result
  return filtered


def plot(kernel, image_path):
  brute = convolve_brute(kernel, image_path)
  fourier = convolve_fft(kernel, image_path)
  f = np.array(color.rgb2gray(io.imread(image_path).astype(float)))
  (rows, columns) = f.shape
  (height, width) = kernel.shape
  clipped  = f[height - 1: rows - height + 1, width - 1:columns - width + 1]
   
  fig, ax = plt.subplots(1,3)
  ax[0].axis('off')
  ax[0].set_title('Original Image')
  ax[0].imshow(clipped, cmap=cm.Greys_r)
  ax[1].set_title('Fourier Convolution')
  ax[1].axis('off')
  ax[1].imshow(np.abs(fourier), cmap=cm.Greys_r)
  ax[2].axis('off')
  ax[2].set_title("Time Domain Convolution")
  ax[2].imshow(brute, cmap = cm.Greys_r)
  plt.show()

if B:
  kernel = np.zeros((5, 5))
  kernel.fill(1/25)
  plot(kernel, "trui.png")

def noise(a0, v0, w0, image_path):
  f = np.array(color.rgb2gray(io.imread(image_path).astype(float)))
  (rows, columns) = f.shape
  for r in range(rows):
    for c in range(columns):
      f[r, c] += a0 * np.cos(v0 * r + w0 * c)
  return f

def trans(f):
  f_fft = fft2(f)
  f_fft_shift = fftshift(f_fft)
  f_fft_abs = np.absolute(f_fft)
  power = f_fft_abs ** 2
  f_logtrans = 100*np.log(1 + power)
  return f_logtrans
 

def denoise(image, v0, w0):
  A = np.zeros(image.shape)
  (rows,columns) = A.shape
  for r in range(rows):
    for c in range(columns):
      A[r, c] = np.cos(v0 * r + w0 * c)
  A_fft = np.abs(fft2(A))
  (r, c) = (np.argmax(np.max(A_fft, axis = 1)), np.argmax(np.max(A_fft, axis = 0)))
  i_fft = fft2(image)
  for i in range(r - 4, r + 5):
    for j in range(c - 4, c + 5):
      i_fft[i, j] = 0
      i_fft[rows - i, columns - j] = 0
  """
  rad = np.sqrt(r ** 2 + c ** 2)
  for i in range(rows):
    for j in range(columns):
      if np.abs(np.sqrt(i**2 + j**2) - rad) <= 2:
        i_fft[i, j] = 0
        i_fft[rows - i - 1, columns - j - 1] = 0
       """
  di = ifft2(i_fft)
  return di






if C:
  f = np.array(color.rgb2gray(io.imread('cameraman.tif').astype(float)))
   
  fig, ax = plt.subplots(2, 2)
  #ax[0][0].imshow(f, cmap=cm.Greys_r)
  #ax[0][1].imshow(trans(f), cmap=cm.Greys_r)
  noisy = noise(100, 12, 12, "cameraman.tif")
  ax[0][0].axis('off')
  ax[0][1].axis('off')
  ax[1][0].axis('off')
  ax[1][1].axis('off')
  ax[0][0].set_title("Noisy Image")
  ax[0][1].set_title("Noisy Spectrum") 
  ax[1][0].set_title("Denoised Image") 
  ax[1][1].set_title("Denoised Spectrum") 
  ax[0][0].imshow(noisy, cmap = cm.Greys_r)
  ax[0][1].imshow(trans(noisy), cmap = cm.Greys_r)
  dn = denoise(noisy, 12, 12)
  ax[1][0].imshow(np.abs(dn), cmap = cm.Greys_r)
  ax[1][1].imshow(trans(dn), cmap = cm.Greys_r)
 
  plt.show() 
  
  

