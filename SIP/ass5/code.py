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
