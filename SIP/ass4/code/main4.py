#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 08:46:04 2019

@author: luise
"""

"""
###############################################################################
          ASSIGNMENT 4 - Group
###############################################################################
"""

#import other files
import functions as fct

#import packages
import os #path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fft2, fftshift, ifft2, ifftshift
#import skimage
import matplotlib
from skimage import io
from skimage import color
from matplotlib import cm
#matplotlib.rcParams['figure.figsize'] = (16,16)
#matplotlib.rcParams['figure.figsize'] = (8, 8)


script_path = os.getcwd().split('\\')[-1]
image_path = script_path +'/images/'

I_trui = np.array(io.imread(image_path + 'trui.png').astype(float))
plt.imshow(I_trui, cmap = cm.gray)
print(I_trui.shape)

### Fourier Transformation

# 1) Convolution with isotropic Gaussian kernel
sigma_arr = [0, 1, 2, 3, 5, 10] 
N = 11

fig, ax = plt.subplots(2, 3, figsize=(9,6))
ax = ax.flatten()
for i in range(len(sigma_arr)):
  if i == 0:
    ax[i].imshow(I_trui, cm.gray)
    ax[i].axis('off')
    ax[i].set_title('original image')
  else:
    I_trui_gauss = fct.scale(I_trui, N, sigma_arr[i])
    ax[i].imshow(I_trui_gauss, cm.gray)
    ax[i].axis('off')
    ax[i].set_title('$\sigma$ = %i' %sigma_arr[i])

fig.savefig(image_path + '1_gauss_sigma.png')


# 3) derivative of a image using FFT
I_trui_derive_x = fct.derive(I_trui, 1, 0)
I_trui_derive_y = fct.derive(I_trui, 0, 1)
I_trui_derive = fct.derive(I_trui, 1, 1)
fig, ax = plt.subplots(1, 4, figsize=(12,3))
ax[0].imshow(I_trui, cm.gray)
ax[0].axis('off')
ax[0].set_title('original image')
ax[1].imshow(I_trui_derive_x, cm.gray)
ax[1].axis('off')
ax[1].set_title('derivative in x')
ax[2].imshow(I_trui_derive_y, cm.gray)
ax[2].axis('off')
ax[2].set_title('derivative in y')
ax[3].imshow(I_trui_derive, cm.gray)
ax[3].axis('off')
ax[3].set_title('derivative in both')

fig.savefig(image_path + '1_derivative.png')







