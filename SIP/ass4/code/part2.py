# Importing packages
import numpy as np
import matplotlib.pyplot as plt
import skimage
import matplotlib
from skimage import io
from skimage import color
from matplotlib import cm

matplotlib.rcParams['figure.figsize'] = (4,4)
from scipy.ndimage.morphology import binary_hit_or_miss
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import black_tophat, white_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, square, rectangle, diamond, ball
f = np.array(
[
[0, 1, 1, 1, 1, 0, 0],
[0, 1, 0, 0, 1, 0, 0],
[0, 1, 0, 0, 1, 0, 0],
[0, 1, 1, 1, 1, 0, 0],
[0, 1, 0, 0, 1, 0, 0],
[0, 1, 0, 0, 1, 0, 0],
[0, 1, 1, 1, 1, 0, 0]
])



#plt.show()

mask1 = np.array(
[
[0, 0, 0],
[1, 0, 1],
[1, 0, 1]
])

mask2 = np.array(
[
[1, 0],
[0, 1]
])

mask3 = np.array([[1, 1, 0]])
mask1 = 1 - mask1
mask2 = 1 - mask2
mask3 = 1 - mask3

imo = [f, binary_dilation(f, mask1), binary_dilation(f, mask2), binary_dilation(f, mask3)]
title = ["Original Image", "Dilation + Mask1", "Dilation + Mask 2", "Dilation + Mask 3"]
fig, ax = plt.subplots(1, 4)
for i in range(4):
  ax[i].imshow(1-imo[i], cmap = "gray")
  #ax[i].axis('off')
  ax[i].set_xticks([])
  ax[i].set_yticks([])
  ax[i].set_title(title[i])

#plt.show()

fig, ax = plt.subplots(1, 3)
titles = ["Mask1 Dilation 3x3", "Mask2 Dilation 2x2", "Mask3 Dilation 1x3"]
masks = [mask1, mask2, mask3]
for i in range(3):
  ax[i].imshow(1 - binary_dilation(masks[i], f), cmap = "gray")
  #ax[i].axis('off')
  ax[i].set_xticks([])
  ax[i].set_yticks([])
  ax[i].set_title(titles[i])
plt.show()

f = color.rgb2gray(io.imread('cells_binary.png', plugin='pil'))
selem = disk(2)
print(selem)
f_closed = binary_opening(f, selem)
f_open  = binary_closing(f, selem)
fig, ax = plt.subplots(1, 3)
ax[0].imshow(f, cmap = "gray")
ax[0].axis('off')
ax[0].set_title("Original Image")
ax[1].imshow(f_open, cmap = "gray")
ax[1].axis('off')
ax[1].set_title("Opening")
ax[2].imshow(f_closed, cmap = "gray")
ax[2].axis('off')
ax[2].set_title("Closing")
plt.show()

def hom(im, B1, B2):
  r1 = binary_erosion(im, B1)
  r2 = binary_erosion((1 - im), B2)
  return r1 & r2

def hom_exp(result, b1):
  (row, col) = result.shape
  count = 0
  other = np.copy(result)
  for r in range(row):
    for c in range(col):
      if result[r, c] == 1:
        count += 1
        print("we got here")
        print(r, r+b1.shape[0], col, col+b1.shape[1])
        print(b1.shape)
        print(other.shape)
        if (other[r:r+b1.shape[0],c:c+b1.shape[1]].shape == b1.shape):
          other[r:r+b1.shape[0],c:c+b1.shape[1]] = b1
  print(count)
  return other
f = color.rgb2gray(io.imread('blobs_inv.png', plugin='pil'))

f = (f >= 10).astype(float)
se1 = np.transpose(np.array([[1, 1, 1, 1, 1]]))
se2 = disk(2)
se3 = np.zeros((3, 3))
se3[1:3,0:2] = 1
print(se3)
hom1 = binary_hit_or_miss(f, se1)
hom2 = binary_hit_or_miss(f, se2)
hom3 = binary_hit_or_miss(f, se3)
hom1 = 1-hom(1-f, se1, 1 - se1)
hom2 = 1-hom(1-f, se2, 1 - se2)
hom3 = 1-hom(1-f, se3, 1- se3)
print(f)
print(1-f)
print(hom3)
what1 =1- white_tophat(1-f, se1)
what2 =1- white_tophat(1-f, se2)
what3 =1- white_tophat(1-f, se3)

bhat1 =1- black_tophat(1-f, se1)
bhat2 =1- black_tophat(1-f, se2)
bhat3 =1- black_tophat(1-f, se3)


fig, ax = plt.subplots(3, 3)
ax[0][0].imshow(hom1, vmin=0,vmax=1,cmap = "gray")
ax[0][0].set_title("Hit Or Miss + Vert Line")
ax[0][1].imshow(hom2, vmin=0,vmax=1,cmap = "gray")
ax[0][1].set_title("Hit Or Miss + Disc")
ax[0][2].imshow(hom3, vmin=0,vmax=1,cmap = "gray")
ax[0][2].set_title("Hit Or Miss + Corner")

ax[1][0].imshow(what1, vmin=0,vmax=1,cmap = "gray")
ax[1][0].set_title("TopHat + Vert Line")
ax[1][1].imshow(what2, vmin=0,vmax=1,cmap = "gray")
ax[1][1].set_title("TopHat + Disc")
ax[1][2].imshow(what3, vmin=0,vmax=1,cmap = "gray")
ax[1][2].set_title("TopHat + Corner")

ax[2][0].imshow(bhat1, vmin=0,vmax=1,cmap = "gray")
ax[2][0].set_title("BottomHat + Vert Line")
ax[2][1].imshow(bhat2, vmin=0,vmax=1,cmap = "gray")
ax[2][1].set_title("BottomHat + Disc")
ax[2][2].imshow(bhat3, vmin=0,vmax=1,cmap = "gray")
ax[2][2].set_title("BottomHat + Corner")

for axi in ax:
  for a in axi:
    a.set_xticks([])
    a.set_yticks([])
plt.show()

def showy(im, a, title):
  a.set_title(title)
  a.imshow(im, cmap = "gray", vmin = 0, vmax = 1)
  a.set_xticks([])
  a.set_yticks([])

fig, ax = plt.subplots(1, 2)
f = color.rgb2gray(io.imread('digits_binary_inv.png', plugin='pil'))
f = (f >= 6000).astype(float)
print(f)
mask = f[5:37, 64:88]
print(mask)
temp = np.zeros(mask.shape)
test = np.ones((3, 3))
result  = hom(1-f, 1-mask, temp)
print(result)
showy(f, ax[0], "Original Image")
#tempy = np.resize(np.copy(mask), (28, 21))
#temps = np.zeros(tempy.shape)
#result_t = hom(1-f, 1-tempy, temps)
 
showy(1-hom_exp(result, 1-mask), ax[1], "Hit or Miss X")
#showy(1-hom_exp(result_t, 1-tempy), ax[2], "Hit or Miss X")
#plt.imshow(1-hom_exp(result, 1-mask), cmap = 'gray')
#plt.imshow(f, cmap = 'gray')
plt.show()
count = 0
for r in result:
  for c in r:
    if c:
      count += 1
print(count)


def showy(im, a, title, cma="gray"):
  a.set_title(title)
  a.imshow(im, cmap = cma, vmin = 0, vmax = 1)
  a.set_xticks([])
  a.set_yticks([])
og = io.imread('bpf.png', plugin='pil')
#1
f = color.rgb2gray(io.imread('bpf.png', plugin='pil'))


import skimage.filters as filters
plt.imshow(og)
plt.show()
plt.imshow(f)
plt.show()
fig, ax = plt.subplots(3, 3)
blurred = filters.gaussian(f, sigma = 4)
dif = f - blurred
thresh = (dif > .07).astype(float)
selem = disk(8)
dil= dilation(thresh, selem)
ero = closing(dil, selem)
#dil= dilation(ero, selem)
#ero = erosion(dil, selem)
clo = closing(ero, disk(20))
showy(og, ax[0][0], "Original Image", None)
showy(f, ax[0][1], "To GreyScale")
showy(blurred, ax[0][2], "Gaussian Filter")
showy(dif, ax[1][0], "Difference")
showy(thresh, ax[1, 1], "Thresholded")
showy(dil, ax[1, 2], "Dilated")
showy(ero, ax[2, 0], "Erosion")
showy(clo, ax[2, 1], "Closing")
results = np.copy(og)
result = f * clo
for i in range(3):
  results[:,:,i] = clo * og[:, :, i]
showy(results, ax[2, 2], "Image + Mask", None)
plt.show()
f = color.rgb2gray(io.imread('money_bin.jpg', plugin='pil'))

f = (f < 10).astype(float)
fig, ax = plt.subplots(2, 3)
showy(1-f, ax[0][0], "Original Image")

selem = disk(15)
op = opening(f, selem)
showy(1-op, ax[0][1], "Phase 1")
op1 = opening(f, disk(18))
showy(1-op1, ax[0][2], "Phase 2")
op2 = opening(f, disk(20))
showy(1-op2, ax[1][0], "Phase 3")
er3 = erosion(f, disk(32))
showy(1-er3, ax[1][1], "Phase 4")
er4  = erosion(f, disk(50))
showy(1-er4, ax[1][2], "Phase 5")
plt.show()
