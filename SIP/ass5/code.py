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


f = np.random.uniform(0, 1, 36)
f = np.resize(f, (6, 6))


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
    
  

kernel = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
fig, ax = plt.subplots(1, 2)
ax[0].imshow(f, vmin = 0, vmax = 1)
ax[1].imshow(right_shift(f), vmin = 0, vmax = 1)
plt.show()
