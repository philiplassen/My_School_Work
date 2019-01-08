import numpy as np
from matplotlib import pyplot as pl

matrix = np.loadtxt(open("projection_matrix.csv", "rb"), delimiter=",")

print(matrix)
print(matrix.shape)

train = np.loadtxt(open("MLWeedCropTrain.csv", "rb"), delimiter=",")
print(train.shape)

labels = train[:, 13]
train  = (train[:,0:13]) 
print(train.shape)

data = np.dot(matrix, train.T).T

print(data.shape)

crop = [data[i, :] for i in range(1000) if labels[i] == 0]
crop1 = [v[0] for v in crop]
crop2 = [v[1] for v in crop]

weed  = [data[i, :] for i in range(1000) if labels[i] == 1]
weed1 = [v[0] for v in weed]
weed2 = [v[1] for v in weed]

print(len(crop))
print(len(weed))
pl.xlabel("Principal Component")
pl.ylabel("Principal Component")
pl.title("Visualizatoin of Weed and Crop Data")
pl.scatter(crop1, crop2)
pl.scatter(weed1, weed2)
pl.show()
print(len(crop1))
print(len(crop2))
#PCA_DATA = np.dot(PROJECTION_MATRIX, TRAIN_DATA).T
