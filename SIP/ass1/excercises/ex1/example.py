import matplotlib.pyplot as pl
import numpy as np
import numpy.random as npr
random_matrix = npr.rand(20, 20)
random_matrix = np.floor(256 * random_matrix)

ims = pl.subplots(1, 2)
im1 = ims.add
ims[0].imshow(random_matrix)
from pylab import ginput
coord = (ginput(1))
coord = (round(coord[0][0]), round(coord[0][1]))
print("You clicked on : " + str(coord))
random_matrix[int(coord[1]),int(coord[0])] = 0
print(coord[0])
ims[1].imshow(random_matrix)
pl.show()

