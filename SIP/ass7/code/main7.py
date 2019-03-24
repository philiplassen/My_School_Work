#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:17:07 2019

@author: luise
"""

"""
###############################################################################
          ASSIGNMENT 7 - Group
###############################################################################
"""

#import other files
import functions as fct

#import packages
import os #path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import histogram
from sklearn.cluster import KMeans




script_path = os.getcwd().split('\\')[-1]
image_path = script_path +'/images/'



"""
EXERCIE 2.1:
  Compute the intensity histograms of the six texture images and visualise 
  these in the report. Also compute the χ 2 -distance between each pair of the 
  six histograms and report them in a table. (Notice that you have to handle 
  the case where both bins in the two histograms are zero.)
  Is it possible to distinguish the six texture classes using intensity histo-
  grams as features of the texture?
"""
img_texture_names = ['canvas1-a-p001.png',
                     'cushion1-a-p001.png',
                     'linseeds1-a-p001.png',
                     'sand1-a-p001.png',
                     'seat2-a-p001.png',
                     'stone1-a-p001.png']

img_texture = [io.imread(image_path + name) for name in img_texture_names]
hist_texture = [fct.norm_hist(histogram(img)) for img in img_texture]
#hist_texture = [fct.norm_hist(np.histogram(img, bins = 256)) for img in img_texture]

#Visualization
fig, ax = plt.subplots(2, 6, figsize=(22,6))
for i in range(len(hist_texture)):
  ax[0, i].imshow(img_texture[i], cmap = 'gray')
  ax[0, i].axis('off')
  ax[0, i].set_title(img_texture_names[i], fontsize = 14)
  #ax[1, i].plot(hist_texture[i][1], hist_texture[i][0])
  ax[1, i].hist(img_texture[i].flatten(), density = True, bins = range(256))
  ax[1, i].set_ylim([0,0.03])
  ax[1, i].set_xlim([0,256])
  ax[1, i].set_xlabel('pixel intensity')
ax[1, 0].set_ylabel('density')

fig.savefig(image_path + '2_histograms.png', bbox = 'tight')

#computing the chi^2 distances
chi2_mat = np.zeros((len(hist_texture), len(hist_texture)))
for r in range(len(hist_texture)):
  for c in range(len(hist_texture)):
    chi2_mat[r, c] = fct.chi2_distance(hist_texture[r], hist_texture[c], 255)

    



"""
EXERCISE 2.2:
  - i) Implement the MR8 filter bank consisting of the oriented Gaussian 
    derivative filters up to order 2 including the zeroth order term (the 
    Gaussian filter itself) and the Laplacian of Gaussian filter. Apply your 
    filter bank implementation to the linseeds1-a-p001.png tex- ture image and 
    illustrate the 8 filter response images for the filter bank.
  - ii) kMeans on the textures
  - iii) texton histograms
  - iv) prediktion
"""


### i) MR8 filter  ############################################################

img_test = io.imread(image_path + 'linseeds1-a-p001.png')
plt.imshow(img_test)

#TESTING the functions
test_filter = fct.Gaussian(20, 20, 0, 0, 5)
plt.imshow(test_filter)
test_response_img = fct.Lxy(img_test, 20, 20, 1, 1, 10)
#test_response_img = fct.Lv(img_test, 20, 20, 10, np.pi)
#test_response_img = fct.Lvv(img_test, 20, 20, 10, np.pi)
#test_response_img = fct.max_Lv(img_test, 20, 20, 10, [0, 1/2 * np.pi, np.pi])
#test_response_img = fct.max_Lvv(img_test, 20, 20, 10, [0, 1/2 * np.pi, np.pi])
plt.imshow(test_response_img)

#The 8 filter response images
sigma_test = 10
ax_test = 6* sigma_test +1
sigma_test_der = [1, 2, 4]
ax_test_der = [6* sigma +1 for sigma in sigma_test_der]
theta_arr_test = [0, np.pi/6, 2*np.pi/6, 3*np.pi/6, 4*np.pi/6, 5*np.pi/6]

img_response_test = fct.MR8_response(img_test, sigma_test, sigma_test_der, theta_arr_test)

fig, ax = plt.subplots(3, 3, figsize=(9,10))
ax = ax.flatten()
for i in range(9):
  ax[i].imshow(img_response_test[: , :, i], cmap = 'gray')
  ax[i].axis('off')
#set titles
ax[0].set_title('original image\n', fontsize = 16)
ax[1].set_title('$L(x, y; \sigma)$\n with $\sigma$ = %i' %sigma_test, fontsize = 14)
ax[2].set_title('$\\nabla^2 L(x, y; \sigma)$\n with $\sigma$ = %i' %sigma_test, fontsize = 14)
for i in range(3):
  ax[3+i].set_title('$L_v(x, y, \sigma)$\n with $\sigma$ = %i' %sigma_test_der[i], fontsize = 14)
for i in range(3):
  ax[6+i].set_title('$max L_vv(x, y, \sigma)$\n with $\sigma$ = %i' %sigma_test_der[i], fontsize = 14)

fig.savefig(image_path + '2_MR8.png')


### ii) kMeans  ###############################################################
 
#reshape X_train 
textures_MR8 = np.array([fct.MR8_response(img, sigma_test, sigma_test_der, theta_arr_test)[:, :, 1:9] for img in img_texture]) #(6, 576, 576, 8)
X = np.zeros((6*576*576, 8))
for i in range(8):
  for n in range(6):
    X_img = textures_MR8[n, :, :, i].flatten()
    number = len(X_img)
    X[n*number : n*number+number, i] = X_img
  
s = 4
#X_small = X[0::s, :]
X_small = X

#Training
kmeans = KMeans(n_clusters=60).fit(X_small)  
labels = kmeans.labels_
labels_whole = kmeans.predict(X)
sum(labels - labels_whole)


### iii) histograms  ##########################################################

number = int(len(labels)/6)
label_arr = np.zeros((6, number))
hist_arr = [0]*6

for i in range (6):
  label_arr[i, :] = labels[i*number : (i+1)*number]
  #label_arr[i, :] = labels_whole[i*number : (i+1)*number]
  hist_arr[i] = np.histogram(label_arr[i, :], bins = 60, density = True)

#Vizualisation
fig, ax = plt.subplots(2, 6, figsize=(22,6))
for i in range(len(hist_texture)):
  ax[0, i].imshow(img_texture[i], cmap = 'gray')
  ax[0, i].axis('off')
  ax[0, i].set_title(img_texture_names[i], fontsize = 14)
  ax[1, i].hist(label_arr[i, :], bins = range(60), density = True)
  ax[1, i].set_ylim([0,0.075])
  ax[1, i].set_xlim([0,60])
  ax[1, i].set_xlabel('predicted label')
ax[1, 0].set_ylabel('density')

fig.savefig(image_path + '2_texton_hist.png')

   
#computing the chi^2 distances
chi2_mat_mr8 = np.zeros((len(hist_arr), len(hist_arr)))
for r in range(len(hist_arr)):
  for c in range(len(hist_arr)):
    chi2_mat_mr8[r, c] = fct.chi2_distance(hist_arr[r], hist_arr[c], 59, fill = False)
 
  
  
### iv) checking mr8 filter  ##################################################

img_check = io.imread(image_path + 'cushion1-a-p012.png')
#computing the MR8 filterbank
texture_MR8_check = fct.MR8_response(img_check, sigma_test, sigma_test_der, theta_arr_test)[:, :, 1:9]
#Run the kMean
X_check = np.zeros((576*576, 8))
for i in range(8):
  X_check[:, i] = texture_MR8_check[:, :, i].flatten()
  
labels_check = kmeans.predict(X_check)

#Vizualisation
hist_check = np.histogram(labels_check, bins = 60, density = True)
fig, ax = plt.subplots(2, 1, figsize=(4,6))
ax[0].imshow(img_check, cmap = 'gray')
ax[0].set_title('cushion1-a-p012.png')
ax[0].axis('off')
ax[1].hist(labels_check, bins = range(60), density = True)
ax[1].set_xlabel('predicted label')
ax[1].set_ylabel('density')

fig.savefig(image_path + '2_texton_hist_check.png')


#chi^2 distances
chi2_check = np.zeros(len(hist_arr))
for r in range(len(hist_arr)):
  chi2_check[r] = fct.chi2_distance(fct.norm_hist(hist_check), fct.norm_hist(hist_arr[r]), 59, fill = False)

np.argmin(chi2_check)




