import numpy as np
from matplotlib import pyplot as pl
from sklearn.cluster import KMeans

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

weed = [data[i, :] for i in range(1000) if labels[i] == 0]
weed1 = [v[0] for v in weed]
weed2 = [v[1] for v in weed]

crop  = [data[i, :] for i in range(1000) if labels[i] == 1]
crop1 = [v[0] for v in crop]
crop2 = [v[1] for v in crop]


print(len(weed))
print(len(crop))
pl.xlabel("Projection on First Principal Component")
pl.ylabel("Projection Second Principal Component")
pl.title("Visualization of Weed and Crop Data")
pl.scatter(weed1, weed2)
pl.scatter(crop1, crop2)

#pl.legend(["Weeds", "Crops"])
#pl.show()

print(labels)

kmeans = KMeans(n_clusters = 2, init = train[0:2, 0:13]).fit(train)
nd = np.dot(matrix, kmeans.cluster_centers_.T).T
print("testing effectiveness")
res = (kmeans.predict(train))
center_1 = nd[0, :]
center_2 = nd[1, :]

pl.scatter([center_1[0]], [center_1[1]])
pl.scatter([center_2[0]], [center_2[1]])


pl.legend(["Weeds", "Crops", "Center 1", "Center 2"])

pl.show()
