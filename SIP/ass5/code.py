"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

XTrain = np.loadtxt('SIPdiatomsTrain.txt', delimiter=',')
XTest = np.loadtxt('SIPdiatomsTest.txt', delimiter=',')
XTrainL = np.loadtxt('SIPdiatomsTrain_classes.txt', delimiter=',')
XTestL = np.loadtxt('SIPdiatomsTest_classes.txt', delimiter=',')

knn = KNeighborsClassifier()
knn.fit(XTrain, XTrainL)

Pred_labels = knn.predict(XTest)
acc = sum(Pred_labels==XTestL) / len(XTestL)
print(acc)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io


np.random.seed(seed=10)
f = np.random.uniform(0, 1, 48)
f = np.resize(f, (9, 9))
f[4, 4] = 1


def shift(f, direction, boundary = "black"):
  (rows, cols) = f.shape
  (rshift, cshift) = direction
  (rshift, cshift) = (int(np.round(rshift)), int(np.round(cshift)))
  val = 0 
  if boundary == "black":
    val = 0
  if boundary == "white":
    val = 1
  max_val = np.max(np.absolute((rshift, cshift)))
  fn = np.pad(f, max_val, "constant", constant_values = val)
  d = np.abs(direction)
  result = np.zeros(f.shape)
  for r in range(rows):
    for c in range(cols):
        result[r, c] = fn[-rshift + r + max_val, -cshift + c + max_val]
  return result



def transform(f, degree, scale, xshift, yshift):
  result = np.zeros(f.shape)
  rows, columns = f.shape
  for r in range(rows):
    for c in range(columns):
      (x, y) = (c, r)
      (x, y) = x - xshift, y - yshift
      (x, y) = (x / scale, y / scale)
      unrot = np.array([[np.cos(degree), -np.sin(degree)], [np.sin(degree), np.cos(degree)]]).T
      temp = np.matmul(unrot, np.array([x, y]))
      (x, y) = temp[0], temp[1]
      (x, y) = int(np.round(x)), int(np.round(y))
      result[r, c] = f[y, x] if x in range(columns) and y in range(rows) else 0
  return result


      
f = np.rot90(color.rgb2gray(io.imread('f.png', plugin='pil')), 2)
print(f.shape)
r = (transform(f, 0, 1, .6, 1.2))
r = (transform(f, np.pi / 8, 0.7, 4.6, 2.7))
#r = (transform(fo, np.pi / 25, 1, 0, 0))


fig, ax = plt.subplots(1, 2)
ax[0].imshow(f, vmin = 0, vmax = 1, cmap = "gray", origin = "lower")
ax[1].imshow(r, vmin = 0, vmax = 1, cmap = "gray", origin = "lower")
ax[0].set_title("Original Image")
ax[1].set_title("Transformed Image")
ax[0].axis('off')
ax[1].axis('off')

plt.show()
