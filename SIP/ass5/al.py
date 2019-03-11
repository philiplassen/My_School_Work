import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import svd
XTrain = np.loadtxt('SIPdiatomsTrain.txt', delimiter=',')
XTest = np.loadtxt('SIPdiatomsTest.txt', delimiter=',')
XTrainL = np.loadtxt('SIPdiatomsTrain_classes.txt', delimiter=',')
XTestL = np.loadtxt('SIPdiatomsTest_classes.txt', delimiter=',')

knn = KNeighborsClassifier()
knn.fit(XTrain, XTrainL)

Pred_labels = knn.predict(XTest)
acc = sum(Pred_labels==XTestL) / len(XTestL)
print(acc)

def center(X):
  Xs = X[:, 0::2]
  Ys = X[:, 1::2]
  print(Xs.shape)
  print(Ys.shape)
  Xn, Yn  =  (Xs - Xs.mean(axis=1).reshape(-1, 1),Ys - Ys.mean(axis=1).reshape(-1, 1))
  result = np.zeros(X.shape)
  result[:, 0::2] = Xs
  result[:, 1::2] = Ys
  return result


points  = center(XTrain)
print(points.shape)

X = np.zeros((2, 90))
X[0, :] = points[0, 0::2]
X[1, :] = points[0, 1::2]

def ralign(X, Ys):
  result = np.zeros(Ys.shape)
  for r in range(Ys.shape[0]):
    points = Ys[r, :]
    Y = np.zeros((2, 90))
    Y[0, :] = points[0::2]
    Y[1, :] = points[1::2]

    (u, s, vh) = svd(np.matmul(Y, X.T), full_matrices=True)
    v = vh.T

    R = v @ u.T
    rp = R @ Y
    s = (X[0, :] @ Y[0, :] + X[1, :] @ Y[1, :]) / (Y[0, :] @ Y[0, :] + Y[1, 0] + Y[1, 0])
    scaled = s * rp
    result[r, 0::2] = scaled[0, :]
    result[r, 1::2] = scaled[1, :]
    print(r)
  return result


test = center(XTest)
testy = test
trainy = points
test = ralign(X, test)
train = ralign(X, points)

knn = KNeighborsClassifier()
knn.fit(train, XTrainL)

Pred_labels = knn.predict(test)
acc = sum(Pred_labels==XTestL) / len(XTestL)
print(acc)


