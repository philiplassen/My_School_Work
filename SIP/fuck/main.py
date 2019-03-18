#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:54:00 2019

@author: luise
"""

"""
###############################################################################
          ASSIGNMENT 6 - Group
###############################################################################
"""

#import other files
import functions as fct

#import packages
import os #path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from pylab import *
from skimage import io
from skimage import feature
from matplotlib import cm


image_path = "Im/"
img_flower = np.array(io.imread(image_path + 'hand.tiff').astype(float))
tau = 5
n = 100

H_map = fct.H(img_flower, tau)
fig, ax = plt.subplots(2, 1, figsize=(4, 6))
ax[0].imshow(img_flower, cmap='gray')
ax[1].imshow(H_map, cmap='gray')

fig.savefig(image_path + '2_blob_detection_t5_H.png')


max_val, min_val = fct.find_max(H_map, n = n)
fig, ax = plt.subplots()
plt.imshow(img_flower, cmap = 'gray')
plt.scatter(max_val[:, 1], max_val[:, 0], marker = 'o', s = .1,  color = 'blue', alpha =1) #marker = 'x', '1'
for i in range(n):
  circle_max = plt.Circle((max_val[i, 1], max_val[i, 0]), 2*tau, color='b', fill = False)
  ax.add_artist(circle_max)
plt.scatter(min_val[:, 1], min_val[:, 0], marker = 'o', s = .1, color = 'red', alpha = 1) #marker = 'x', '1'
fig.savefig(image_path + '2_blob_detection_t5_blob.png')


tau_arr = [1, 2, 5, 10, 20, 50]
n = 100
#tau_arr = [1, 2, 1, 2, 1, 2]
fig, ax = plt.subplots(2, 3, figsize=(24, 12))
ax = ax.flatten()
for i in range(6):
  H_map_test = fct.H(img_flower, tau_arr[i])
  max_val_test, min_val_test = fct.find_max(H_map_test, n = n)
  ax[i].imshow(img_flower, cmap = 'gray')
  ax[i].scatter(max_val_test[:, 1], max_val_test[:, 0], marker = '.', s = .1, color = 'blue', alpha =1) #marker = 'x', '1'
  for j in range(n):
    circle_max = plt.Circle((max_val_test[j, 1], max_val_test[j, 0]), 2*tau_arr[i], color='b', fill = False)
    ax[i].add_artist(circle_max)

  fig.savefig(image_path + '2_blob_detection_test_100.png')


tau_arr = [1, 2, 5, 10, 20, 50]
n = 20
#tau_arr = [1, 2, 1, 2, 1, 2]
fig, ax = plt.subplots(2, 3, figsize=(24, 12))
ax = ax.flatten()
for i in range(6):
  H_map_test = fct.H(img_flower, tau_arr[i])
  max_val_test, min_val_test = fct.find_max(H_map_test, n = n)
  ax[i].imshow(img_flower, cmap = 'gray')
  ax[i].scatter(max_val_test[:, 1], max_val_test[:, 0], marker = '.', s = .1, color = 'blue', alpha =1) #marker = 'x', '1'
  for j in range(n):
    circle_max = plt.Circle((max_val_test[j, 1], max_val_test[j, 0]), 2*tau_arr[i], color='b', fill = False)
    ax[i].add_artist(circle_max)

  ax[i].scatter(min_val_test[:, 1], min_val_test[:, 0], marker = '.', s = .1, color = 'red', alpha = 1) #marker = 'x', '1'
  ax[i].set_title('$\\tau$ = %i' %tau_arr[i], fontsize = 15)

fig.savefig(image_path + '2_blob_detection_test_20.png')



tau_arr = [3, 5, 7, 10, 15, 20, 25, 50]
#tau_arr = [1, 2,1, 3]
n = 100
fig, ax = plt.subplots(figsize = (12, 9))
ax.imshow(img_flower, cmap = 'gray')
for i in range(len(tau_arr)):
  H_map_test = fct.H(img_flower, tau_arr[i])
  max_val_test, min_val_test = fct.find_max(H_map_test, n = n)
  ax.scatter(max_val_test[:, 1], max_val_test[:, 0], marker = '.', s = .1, color = 'blue', alpha =1) #marker = 'x', '1'
  for j in range(n):
    circle_max = plt.Circle((max_val_test[j, 1], max_val_test[j, 0]), 1.8*tau_arr[i], color='b', fill = False)
    ax.add_artist(circle_max)

  ax.scatter(min_val_test[:, 1], min_val_test[:, 0], marker = '.', s = .1, color = 'red', alpha = 1) #marker = 'x', '1'
  fig.savefig(image_path + '2_blob_detection_test_20compact.png')






tau_arr = [3, 5, 7, 10, 12, 15, 20, 25, 30]
#tau_arr = [1, 2, 3]
H_map_test, ind_tau = fct.H_multitau(img_flower, tau_arr)
n = 100  #max = 421500
max_val_test, min_val_test = fct.find_local_max(H_map_test, n = n, k=1)

fig, ax = plt.subplots(figsize = (12, 9))
ax.imshow(img_flower, cmap = 'gray')
ax.scatter(max_val_test[:, 1], max_val_test[:, 0], marker = '.', s = .1, color = 'blue', alpha =1) #marker = 'x', '1'
for j in range(n):
  circle_max = plt.Circle((max_val_test[j, 1], max_val_test[j, 0]), 1.8*tau_arr[ind_tau[int(max_val_test[j, 0]), int(max_val_test[j, 1])]], color='b', fill = False)
  ax.add_artist(circle_max)

ax.scatter(min_val_test[:, 1], min_val_test[:, 0], marker = '.', s = .1, color = 'red', alpha = 1) #marker = 'x', '1'
fig.savefig(image_path + '2_blob_detection_test_20compact_multi.png')








