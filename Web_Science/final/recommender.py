#!/usr/bin/env python3
import pandas as pd
import sys

VERBOSE = False

if '-v' in sys.argv:
  VERBOSE = True

def log(message):
  if VERBOSE:
    print(message)

folder = "ml-latest-small/"




linksdf = pd.read_csv(folder + "links.csv")
moviesdf = pd.read_csv(folder + "movies.csv")
tagsdf = pd.read_csv(folder + "tags.csv")
ratingsdf = pd.read_csv(folder + "ratings.csv")

log("Links headers are : " + str(list(linksdf)))
log("Tags headers are : " + str(list(tagsdf)))
log("Movies headers are : " + str(list(moviesdf)))
log("Ratings headers are : " + str(list(ratingsdf)))


"""
df = pd.read_csv("crowd.tsv", encoding = "iso-8859-1", sep = '\t')
df = df[['id', 'question', 'opinion']]
mean = df.groupby(['id'], as_index = False, sort = False).mean().rename(columns = {'opinion' : 'opinion_mean'})[['id', 'opinion_mean']]
Opinions = pd.merge(df, mean, on = 'id', how = 'left', sort = False)
Opinions['adjusted'] = Opinions['opinion'] - Opinions['opinion_mean']
result = pd.DataFrame({'id':Opinions['id'], "question": Opinions['question'], "opinion" : Opinions['adjusted']})
result1 = result.pivot_table(index = 'id', columns = 'question', values = 'opinion').fillna(0)
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(result1)
resulty = np.argsort(cosine_similarity(result1))
user = int(input("What is your ID? : "))
k = int(input("Value for K cut off? : "))
print(np.flip(resulty[user, -(k+1): 37]))
"""

