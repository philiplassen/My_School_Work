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
from matplotlib import pyplot as pl

trainInput = np.loadtxt(open("trainInput.csv", "rb"), delimiter=",")
trainTarget = np.loadtxt(open("trainTarget.csv", "rb"), delimiter=",")
tf = pd.read_csv("trainTarget.csv", names=["target"])


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Standardizing data
trainInput  = StandardScaler().fit_transform(trainInput)

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(trainInput)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, tf[['target']]], axis = 1)



fig = pl.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)


targets = [0, 1]
colors = ['r', 'b']

print(finalDf['target'])
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()



pl.show()



