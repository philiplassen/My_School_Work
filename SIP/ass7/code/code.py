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
noise = np.random.normal(0, 1, f.shape) / 10
#noise.clip(0, 1)
#kernel = gaussian_filter(np.zeros((5, 5)), 1) / 50
kernel = np.ones((1, 1))
kernel = np.zeros((5, 5))
kernel.fill(1/25)
print(kernel.shape)

"""
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
"""
g_kernel = np.array([
[0.023528,	0.033969,	0.038393, 0.033969,	0.023528],
[0.03396, 	0.049045,	0.055432, 0.049045,	0.033969],
[0.03839, 	0.055432,	0.062651, 0.055432,	0.038393],
[0.03396, 	0.049045,	0.055432, 0.049045,	0.033969],
[0.02352, 	0.033969,	0.038393, 0.033969,	0.023528]])

g_kernel = np.array([
[0.016641,	0.018385,	0.019518,	0.019911,	0.019518,	0.018385,	0.016641],
[0.018385,	0.020312,	0.021564,	0.021998,	0.021564,	0.020312,	0.018385],
[0.019518,	0.021564,	0.022893,	0.023354,	0.022893,	0.021564,	0.019518],
[0.019911,	0.021998,	0.023354,	0.023824,	0.023354,	0.021998,	0.019911],
[0.019518,	0.021564,	0.022893,	0.023354,	0.022893,	0.021564,	0.019518],
[0.018385,	0.020312,	0.021564,	0.021998,	0.021564,	0.020312,	0.018385],
[0.016641,	0.018385,	0.019518,	0.019911,	0.019518,	0.018385,	0.016641]])

#g_kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]]) / 16
"""
def degrade(kernel, image, image_noise):
  f = image
  f_fft = fft2(f)
  k_fft = fft2(kernel, shape = f.shape)
  i_fft = fft2(image_noise, shape = f.shape)
  result = ifft2((f_fft * k_fft) + i_fft)
  return result
"""

def inverse_filter(kernel, degraded_image, n):
  g = degraded_image
  h = kernel
  (rows, columns) = g.shape
  (height, width) = kernel.shape
  g_fft = fft2(g)
  h_fft = fft2(kernel, g_fft.shape)
  f_fft = g_fft / h_fft
  #f_fft = np.clip(f_fft, n / 10, np.max(f_fft))
  result = ifft2(f_fft)
  return result
  #return result[height - 1: rows - height + 1, width - 1:columns - width + 1]
  

def degrade(kernel, image, image_noise):
  f = image
  f_fft = fft2(f)
  k_fft = fft2(kernel, shape = f.shape)
  i_fft = fft2(image_noise, shape = f.shape)
  result = ifft2((f_fft * k_fft) + i_fft)
#  result = ifft2(f_fft * k_fft)
#  print(np.max(result), np.min(result))
#  result = result + 1 * image_noise
  return result

def wiener_filter(kernel, degraded_image, K):
  g = degraded_image
  h = kernel
  G = fft2(g)
  H = fft2(kernel, G.shape)
  result = (1 / H) * G * np.square(np.absolute(H)) / (np.square(np.absolute(H)) + K)
  return ifft2(result)

def weiner_filter(kernel, degraded_image, K):
  g = degraded_image
  h = kernel
  G = fft2(g)
  H = fft2(kernel, G.shape)
  result = (1 / H) * G * np.square(np.absolute(H)) / (np.square(np.absolute(H)) + K)
  return ifft2(result)


n0 = np.zeros(f.shape)
n1 = np.random.normal(0, 1, f.shape) / 10
n2 = np.random.normal(0, 3, f.shape) / 10 
a_kernel = kernel
def d_plot():
  n0 = np.zeros(f.shape)
  n1 = np.random.normal(0, 1, f.shape) / 10
  n2 = np.random.normal(0, 3, f.shape) / 10 
  a_kernel = kernel
  i_kernel = [[1]]
  fig, ax = plt.subplots(2, 3)
  for a in ax:
   for b in a:
     b.set_axis_off()
  ax[0][0].imshow(f, cmap = "gray")
  ax[0][0].set_title("Original Image")
  ax[0][1].imshow(np.abs(degrade(a_kernel, f, n0)), cmap = "gray")
  ax[0][1].set_title("Averaging Filter")
  ax[0][2].imshow(np.abs(degrade(a_kernel, f, n2 * 100)), cmap = "gray")
  ax[0][2].set_title("With Noise")
  
  ax[1][0].imshow(f, cmap = "gray")
  ax[1][0].set_title("Original Image")
  ax[1][1].imshow(np.abs(degrade(g_kernel, f, n0)), cmap = "gray")
  ax[1][1].set_title("Gaussian Filter")
  ax[1][2].imshow(np.abs(degrade(g_kernel, f, n2 * 100)), cmap = "gray")
  ax[1][2].set_title("With Noise")
  """
  ax[1][0].imshow(f, cmap = "gray")
  ax[1][1].imshow(np.abs(degrade(g_kernel, f, n0)), cmap = "gray")
  ax[1][2].imshow(np.abs(degrade(g_kernel, f, n2 * 100)), cmap = "gray")
  """
  plt.show()


n0 = np.zeros(f.shape)
n1 = np.random.normal(0, 1, f.shape) / 10
n2 = np.random.normal(0, 3, f.shape) / 10 
a_kernel = kernel
 
d1 = np.abs(degrade(a_kernel, f, n0 ))
d2 = np.abs(degrade(a_kernel, f, n1))
d3 = np.abs(degrade(g_kernel, f, n0))
d4 = np.abs(degrade(g_kernel, f, n1 ))
def i_plot():
  n0 = np.zeros(f.shape)
  n1 = np.random.normal(0, 1, f.shape) / 10
  n2 = np.random.normal(0, 3, f.shape) / 10 
  a_kernel = kernel
  i_kernel = [[1]]
  fig, ax = plt.subplots(2, 3)
  for a in ax:
   for b in a:
     b.set_axis_off()
  (mi, ma) = (np.min(np.abs(inverse_filter(a_kernel, d1, .001))), np.max(np.abs(inverse_filter(a_kernel, d1, 0.001))))
  ax[0][0].imshow(d1, cmap = "gray")
  ax[0][0].set_title("Degraded Image with Averaging")
  ax[0][1].imshow(np.abs(inverse_filter(a_kernel, d1, 0.001)), cmap = "gray", vmin = mi, vmax = 265)  
  ax[0][1].set_title("Recovered Image")
  ax[0][2].imshow(np.abs(inverse_filter(a_kernel, d2, 0.001)), cmap = "gray", vmin = mi, vmax = 265)
  ax[0][2].set_title("Recovered Image with Noise")
  ax[1][0].imshow(d3, cmap = "gray")
  ax[1][0].set_title("Degraded Image with Gaussian")
  ax[1][1].imshow(np.abs(inverse_filter(g_kernel, d3, 0.001)), cmap = "gray", vmin = mi, vmax = 265)  
  ax[1][1].set_title("Recovered Image")
  ax[1][2].imshow(np.abs(inverse_filter(g_kernel, d4, 0.001)), cmap = "gray", vmin = mi, vmax = 265)
  ax[1][2].set_title("Recovered Image with Noise")
  
  plt.show()

def w_plot():
  n0 = np.zeros(f.shape)
  n1 = np.random.normal(0, 1, f.shape) / 10
  n2 = np.random.normal(0, 3, f.shape) / 10 
  a_kernel = kernel
  i_kernel = [[1]]
  fig, ax = plt.subplots(2, 3)
  for a in ax:
   for b in a:
     b.set_axis_off()
  (mi, ma) = (np.min(np.abs(wiener_filter(a_kernel, d1, .001))), np.max(np.abs(inverse_filter(a_kernel, d1, 0.001))))
  ax[0][0].imshow(d1, cmap = "gray")
  ax[0][0].set_title("Degraded Image with Averaging")
  ax[0][1].imshow(np.abs(wiener_filter(a_kernel, d1, 0.001)), cmap = "gray", vmin = mi, vmax = 265)  
  ax[0][1].set_title("Recovered Image")
  ax[0][2].imshow(np.abs(wiener_filter(a_kernel, d2, 0.001)), cmap = "gray", vmin = mi, vmax = 265)
  ax[0][2].set_title("Recovered Image with Noise")
  ax[1][0].imshow(d3, cmap = "gray")
  ax[1][0].set_title("Degraded Image with Gaussian")
  ax[1][1].imshow(np.abs(wiener_filter(g_kernel, d3, 0.001)), cmap = "gray", vmin = mi, vmax = 265)  
  ax[1][1].set_title("Recovered Image")
  ax[1][2].imshow(np.abs(wiener_filter(g_kernel, d4, 0.001)), cmap = "gray", vmin = mi, vmax = 265)
  ax[1][2].set_title("Recovered Image with Noise")
  
  plt.show()

"""
def w_plot():
  n0 = np.zeros(f.shape)
  n1 = np.random.normal(0, 1, f.shape) / 10
  n2 = np.random.normal(0, 3, f.shape) / 10 
  a_kernel = kernel
  i_kernel = [[1]]
  fig, ax = plt.subplots(2, 3)
  ax[0][0].imshow(f, cmap = "gray")
  ax[0][1].imshow(np.abs(weiner_filter(a_kernel, d1, 0.001)), cmap = "gray")
  ax[0][2].imshow(np.abs(weiner_filter(a_kernel, d2, 0)), cmap = "gray")
  ax[1][0].imshow(f, cmap = "gray")
  ax[1][1].imshow(np.abs(weiner_filter(g_kernel, d3, 0)), cmap = "gray")
  ax[1][2].imshow(np.abs(weiner_filter(g_kernel, d4, 0)), cmap = "gray")
  plt.show()



def weiner_filter(kernel, degraded_image, K):
  g = degraded_image
  h = kernel
  G = fft2(g)
  H = fft2(kernel, G.shape)
  result = (1 / H) * G * np.square(np.absolute(H)) / (np.square(np.absolute(H)) + K)
  return ifft2(result)
"""
d_plot()
"""
degraded_image = (degrade(kernel, f, noise))
#degraded_image = (degrade(kernel, f, np.zeros(f.shape)))
restored_image = inverse_filter(kernel, degraded_image, .001)
#restored_image = (weiner_filter(kernel, degraded_image,0))
fig, ax = plt.subplots(1, 3)

ax[0].imshow(np.abs(degrade(kernel, f, noise)), vmin = 0, vmax = 255, cmap = 'gray')
ax[1].imshow(np.abs(restored_image), vmin = 0, vmax = 255, cmap = 'gray')
ax[2].imshow(f, vmin = 0, vmax = 255, cmap = 'gray')
plt.show()
"""
