import sys
DEBUG = False
KMeans = False
for arg in sys.argv:
  if arg == '-v':
    DEBUG = True
  if arg == '-k':
    KMeans = True

#Verbose for Debugging
def log(message):
  if DEBUG:
    print(message)



import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

trainInput = np.loadtxt(open("trainInput.csv", "rb"), delimiter=",")
trainTarget = np.loadtxt(open("trainTarget.csv", "rb"), delimiter=",")
tf = pd.read_csv("trainTarget.csv", names=["target"])



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


pl.plot(decomp.explained_variance_)
pl.title("Eigenspectrum")
pl.xlabel("Number of Eigenvector")
pl.ylabel("EigenValue")
pca = PCA(n_components = 2)
PCs = pca.fit_transform(trainInput)
tempFrame = pd.DataFrame(data = PCs, columns = ['principal component 1', 'principal component 2'])
completedFrame = pd.concat([tempFrame, tf[['target']]], axis = 1)



fig = pl.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)


targets = [0, 1]
colors = ['r', 'b']

print(completedFrame['target'])
for target, color in zip(targets,colors):
    indicesToKeep = completedFrame['target'] == target
    ax.scatter(completedFrame.loc[indicesToKeep, 'principal component 1']
               , completedFrame.loc[indicesToKeep, 'principal component 2'], c = color, s = 25)
ax.legend(targets)
ax.grid()
ax.set_title("Data Projected on first 2 Principal Components")
def getIndex(label):
  index = 0
  while (trainTarget[index] != label):
    index += 1
  return index
if KMeans == True:
  index0 = getIndex(0)
  index1 = getIndex(1)

  first0 = trainInput[index0, :]
  first1 = trainInput[index1, :]

  from sklearn.cluster import KMeans

  kmeans = KMeans(n_clusters = 2, init = trainInput[[index0, index1], :]).fit(trainInput)
  projections = pca.transform(kmeans.cluster_centers_)
  print(projections)
  [x, y] = projections[0, :]
  ax.scatter([x], [y], c = "g", s = 200)
  [x, y] = projections[1, :]
  ax.scatter([x], [y], c = "orange", s = 200)
  ax.legend(targets + ["K Means Center", "K Means Center"])
  ax.set_title("Cluster Center Projections")
pl.show()


fileX = "trainSubsetInput.csv"
filey = "trainSubsetTarget.csv"


X = np.loadtxt(open("trainSubsetInput.csv", "rb"), delimiter=",")
y = np.loadtxt(open("trainSubsetTarget.csv", "rb"), delimiter=",")
testX = np.loadtxt(open("testInput.csv", "rb"), delimiter=",")
testy = np.loadtxt(open("testTarget.csv", "rb"), delimiter=",")



index0s = [i for i in range(len(y)) if y[i] == 0]
index1s = [i for i in range(len(y)) if y[i] == 1]

log(len(index0s))
log(len(index1s))
log(len(y))

from sklearn import svm
from sklearn.model_selection import GridSearchCV


# Need a method for generating G given X and y
def G(X, y):
  index0s = [i for i in range(len(y)) if y[i] == 0]
  index1s = [i for i in range(len(y)) if y[i] == 1]
  X0 = X[index0s, :]
  X1 = X[index1s, :]
  

  G = []
  for i in range(len(y)):
    G += [shortestDistance(X[i, :], X1)] if y[i] == 0 else [shortestDistance(X[i, :], X0)]
  log(G[len(G) - 1])
  log(G[0])
  return np.median(G)


def shortestDistance(x, X):
  distances = [np.linalg.norm(x-xi) for xi in X]
  distances.sort()
  return distances[0]

theta = G(X, y)
gamma = 1.0 / (2 * (theta ** 2))

print("Value of theta : " + str(theta))
print("Value of gamma : " + str(gamma))

b = 2


gammas = [gamma * (b ** i) for i in range(-3, 4)]
Cs = [b ** i for i in range(-1, 4)]
param_grid = {'C' : Cs, 'gamma' : gammas}
log("Paramaters under consideration")
log("C values are : " + str(Cs))
log("Gamma values are : " + str(gammas))
def params(X, y, nfolds, param_grid):
  grid_search = GridSearchCV(svm.SVC(kernel = 'rbf'), param_grid, cv = nfolds)
  grid_search.fit(X, y)
  grid_search.best_params_
  return grid_search.best_params_

log("Selected Paramater are ....")
log("---------------------------")
params = params(X, y, 5, param_grid)
log(params)
log("---------------------------")

log("Using paramaters to train model...")
model = svm.SVC(C = params['C'], gamma = params['gamma'], kernel = 'rbf')
model.fit(X, y)
log("Model has been trained")
log("Testing model on test data....")
log("-----------------------------")
accuracy = model.score(testX, testy)
log(accuracy)
log("-----------------------------")
log("Testing model on train  data....")
log("-----------------------------")
accuracy = model.score(X, y)
log(accuracy)
log("-----------------------------")

