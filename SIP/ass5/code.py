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


f = np.random.uniform(0, 1, 48)
f = np.resize(f, (6, 8))


def shift(f, direction, boundary = "black"):
  (rows, cols) = f.shape
  val = 0 
  if boundary == "black":
    val = 0
  if boundary == "white":
    val = 1
  fn = np.pad(f, np.absolute(direction), "constant", constant_values = val)
  result = np.zeros(f.shape)
  print(np.round(fn, 1))
  print(np.round(f, 1))
  for r in range(rows):
    for c in range(cols):
        result[r, c] =  fn[r+direction, c] if (direction > 0) else fn[r + np.abs(direction), c + 2 * np.abs(direction)]
  return result







def right_shift(f, boundary = "black"):
  (rows, cols) = f.shape
  val = 0 
  if boundary == "black":
    val = 0
  if boundary == "white":
    val = 1
  fn = np.pad(f, 1, "constant", constant_values = val)
  result = np.zeros(f.shape)
  for r in range(rows):
    for c in range(cols):
      result[r, c] = fn[r+1, c]
  return result
    
  


fig, ax = plt.subplots(1, 2)
ax[0].imshow(f, vmin = 0, vmax = 1)
ax[1].imshow(shift(f, -3), vmin = 0, vmax = 1)
plt.show()
