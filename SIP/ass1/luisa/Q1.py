#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:13:35 2019

@author: luise
"""


"""
###############################################################################
Question 1:
  Write a program to display a random 20-by-20 pixel gray scale image. In the 
  figure, make sure that the axes are labelled ‘X’ and ‘Y’, the tick marks are 
  visible and also labelled 0-19. Use the Python function ginput (see Example 
  1.5) to record the selection of a pixel using the mouse, and display that 
  pixel’s X, Y co-ordinates, and then change the colour of the selected pixel 
  to black.
###############################################################################
"""

import matplotlib.pyplot as plt
import numpy as np
from pylab import ginput

  
def black_pixel(save_path):
  
  I = np.random.rand(20, 20)*255
  J = I
  
  plt.figure()
  plt.subplot(1,2,1) #3 rows, 1 column, 1st image
  plt.imshow(I, cmap = 'gray')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.xticks(np.arange(0, 20))
  plt.yticks(np.arange(0, 20))
  
  print('Click on one point in the image')
  coord = ginput(1)
  print('You clicked: ' + str(coord))
  J[int(round(coord[0][1])), int(round(coord[0][0]))] = 0
  #J[int(round(coord[1][0])), int(round(coord[1][1]))] = 0
  #J[int(round(coord[2][0])), int(round(coord[2][1]))] = 0
  
  plt.subplot(1,2,2)
  plt.imshow(J, cmap = 'gray')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.xticks(np.arange(0, 20))
  plt.yticks(np.arange(0, 20))
  plt.show() 
  plt.savefig(save_path + 'Q1_black_pixel.png')

  

