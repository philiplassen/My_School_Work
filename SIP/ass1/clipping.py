import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
im1 = imread("Images/cameraman.tif")
im2 = imread("Images/cell.tif")
im1 = resize(im1, im2.shape)
im1 = np.round(im1 * 255).astype(int)
result_plus  = np.clip(im1.astype('uint16') + im2, 0, 255).astype('uint8')
result_minus1 =  np.clip(im1.astype('uint16') - im2, 0, 255).astype('uint8')
result_minus2 =  np.clip(im2.astype('uint16') - im1, 0, 255).astype('uint8')

print(im1)
print(im2)
plt.imshow(result_plus, cmap = "gray")
plt.imshow(result_minus1, cmap = "gray")
plt.imshow(result_minus2, cmap = "gray")
plt.show()
