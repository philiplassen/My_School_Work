import matplotlib.pyplot as plt
import skimage.io as ski
from pylab import ginput
import numpy as np
import numpy.random as npr


def display_pixels():
    I = np.random.rand(20, 20)*255
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
    I[int(round(coord[0][1])), int(round(coord[0][0]))] = 0
    plt.subplot(1,2,2)
    print("we are here")
    plt.imshow(I , cmap = 'gray')
    print("on the other side")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(np.arange(0, 20))
    plt.yticks(np.arange(0, 20))
    plt.show()

display_pixels()

