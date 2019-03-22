#!/usr/bin/env python3

"""
############################################################

Assignment 7 - Group

############################################################
"""
import numpy as np
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter
from skimage import color
from skimage import io
import matplotlib.pyplot as plt

f = np.array(color.rgb2gray(io.imread('trui.png').astype(float)))
noise = np.random.normal(0, 1, f.shape)
noise.clip(0, 1)
#kernel = gaussian_filter(np.zeros((5, 5)), 1) / 50
kernel = np.ones((1, 1))
kernel = np.zeros((5, 5))
kernel.fill(1/25)
print(kernel.shape)


def degrade(kernel, image, image_noise):
  f = image
  (rows, columns) = f.shape
  (height, width) = kernel.shape
  f_fft = fft2(f)
  k_fft = fft2(kernel, shape = f.shape)
  i_fft = fft2(image_noise, shape = f.shape)
  result = ifft2((f_fft * k_fft) + i_fft)
  return result
  #return result[height - 1: rows - height + 1, width - 1:columns - width + 1]

def inverse_filter(kernel, degraded_image, n):
  g = degraded_image
  h = kernel
  (rows, columns) = g.shape
  (height, width) = kernel.shape
  g_fft = fft2(g)
  h_fft = fft2(kernel, g_fft.shape)
  f_fft = g_fft / h_fft
  f_fft = np.clip(f_fft, n, np.max(f_fft))
  result = ifft2(f_fft)
  return result
  #return result[height - 1: rows - height + 1, width - 1:columns - width + 1]
  

degraded_image = (degrade(kernel, f, noise))
degraded_image = (degrade(kernel, f, np.zeros(f.shape)))
restored_image = (inverse_filter(kernel, degraded_image, 0.01))
fig, ax = plt.subplots(1, 3)

ax[0].imshow(np.abs(degrade(kernel, f, noise)), vmin = 0, vmax = 255, cmap = 'gray')
ax[1].imshow(np.abs(restored_image), vmin = 0, vmax = 255, cmap = 'gray')
ax[2].imshow(f, vmin = 0, vmax = 255, cmap = 'gray')
plt.show()
