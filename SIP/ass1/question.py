import matplotlib.pyplot as plt
import numpy as np
from pylab import ginput

  
def black_pixel():
  
  I = np.random.rand(20, 20)*255
  
  plt.figure()
  plt.subplot(1,2,1) 
  plt.imshow(I, cmap = 'gray')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.xticks(np.arange(0, 20))
  plt.yticks(np.arange(0, 20))
  
  print('Click on one point in the image')
  coord = ginput(1)
  (row, column)  = (int(round(coord[0][1])), int(round(coord[0][0])))
  print('You clicked on ((row : %s), (column : %s))' % (row, column))
  updated = I
  updated[row, column] = 0
  plt.subplot(1,2,2)
  plt.imshow(updated, cmap = 'gray')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.xticks(np.arange(0, 20))
  plt.yticks(np.arange(0, 20))
  plt.show()

black_pixel()
