import sys
DEBUG = False
for arg in sys.argv:
  if arg == '-v':
    DEBUG = True

#Verbose for Debugging
def log(message):
  if DEBUG:
    print(message)



import numpy as np
from matplotlib import pyplot as pl


# Question 1 Starts here

trainInput = np.loadtxt(open("trainInput.csv", "rb"), delimiter=",")
trainTarget = np.loadtxt(open("trainTarget.csv", "rb"), delimiter=",")

log(trainInput.shape)
log(trainTarget.shape)

#Labels are 1 and 0 we can take the sum for frequency 1 labeled data
lengthOfData = trainTarget.shape[0]
countLabel1 = sum(trainTarget)
countLabel0 = lengthOfData - countLabel1

print("Output for Question 1")
print("")
print("Frequency of label 0 : " + str(countLabel0 / lengthOfData))
print("Frequency of label 1 : " + str(countLabel1 / lengthOfData))
print("")

#Question 2
from sklearn.decomposition import PCA

decomp = PCA()
decomp.fit(trainInput)
count = 0
limit = 0.9
value = 0.0
while (value < limit):
  value += decomp.explained_variance_ratio_[count]
  count += 1

log(value)
log(count)

print("The number of Principal Components needed to explain 90% of the variance : " + str(count))
results = np.matmul(trainInput, decomp.components_[:, 0:2])

new_pca = PCA(n_components=2)
new_pca.fit(trainInput)
results = new_pca.transform(trainInput)



c1x = [results[i][0] for i in range(lengthOfData) if trainTarget[i] == 1]
c2x = [results[i][0] for i in range(lengthOfData) if trainTarget[i] == 0]
c1y = [results[i][1] for i in range(lengthOfData) if trainTarget[i] == 1]
c2y = [results[i][1] for i in range(lengthOfData) if trainTarget[i] == 0]



log(len(c1x))
log(len(c2x))
log(len(c1y))
log(len(c2y))
log(c1x[0:5])
log(c2x[0:5])
log(c1y[0:5])
log(c2y[0:5])
pl.subplot(2, 1, 1)
pl.xlabel("Singular Value")
pl.plot(decomp.singular_values_)
#pl.scatter([i for i in range(len(decomp.singular_values_))], decomp.singular_values_)


pl.subplot(2, 1, 2)

pl.scatter(c1x, c1y)
pl.scatter(c2x, c2y)

pl.show()


#Question 3 (Clustering)


#getting Index of first 0
def getIndex(label):
  index = 0
  while (trainTarget[index] != label):
    index += 1
  return index

index0 = getIndex(0)
index1 = getIndex(1)

first0 = trainInput[index0, :]
first1 = trainInput[index1, :]

log("Testing initial centers")
log("index 0 : " + str(index0))
log("index 0 : " + str(index1))
log("Shapes of centers : " + str(first0.shape) + " " + str(first1.shape))

from sklearn.cluster import KMeans

initial_center = np.vstack(first0, first1)
cluster = KMeans(n_clusters = 2, init = initial_center).fit(trainInput)





