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
import pandas as pd

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

print(theta)
print(gamma)

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
