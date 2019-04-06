#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer as vectorizer




N = 30

VERBOSE = False

if '-v' in sys.argv:
  VERBOSE = True

def log(message):
  if VERBOSE:
    print(message)


"""
Importing data from different Folders
"""
folder = "ml-latest-small/"

linksdf = pd.read_csv(folder + "links.csv")
moviesdf = pd.read_csv(folder + "movies.csv")
tagsdf = pd.read_csv(folder + "tags.csv")
ratingsdf = pd.read_csv(folder + "ratings.csv")

"""
Logging headers from the different DataFrames
"""
log("Links headers are : " + str(list(linksdf)))
log("Tags headers are : " + str(list(tagsdf)))
log("Movies headers are : " + str(list(moviesdf)))
log("Ratings headers are : " + str(list(ratingsdf)))


def combine(val, arr):
  result = val
  for a in arr:
    result += (" " + a)
  return result

dft = tagsdf
dfm = moviesdf
mids = dfm.movieId.unique()
dold = {mid : dfm.loc[dfm['movieId'] == mid].genres.item() for mid in mids}

d = {mid : combine(dold[mid], list(dft.loc[dft['movieId'] == mid].tag)) for mid in mids}
dic = np.array(list(d.items()))
keys = dic[:, 0]
vals = dic[:, 1]
vect = vectorizer()

vect.set_params(stop_words='english')
# include 1-grams and 2-grams
print("Making vocabulary")
vect.set_params(ngram_range=(1, 1))
X = vect.fit_transform(vals)
print("Vocabulary Made")
print(vect.vocabulary_)



def similarity(pivot_table):
  return cosine_similarity(pivot_table)  
print(X.toarray().shape)
Xa = X.toarray()
cm = similarity(Xa)

def rowToMid(row):
  return mids[row]

N = 20
correlation_matrix = cm
weight_matrix = np.flip(np.sort(correlation_matrix), axis = 1)
index_matrix = np.flip(np.argsort(correlation_matrix), axis = 1)
dictionary = {} 

for idx in range(correlation_matrix.shape[0]):
    cmids  = [mids[i] for i in index_matrix[idx, 1:N+1]]
    p2 = weight_matrix[idx, 1:N + 1]
    key = mids[idx] 
    dictionary[key] = (cmids, p2)

def mean_id(data):
  mean_user_rating = data.groupby(['userId'], as_index = False, sort = False).mean().rename(
  columns = {'rating' : 'rating_mean'})[['userId', 'rating_mean']]
  return mean_user_rating

meandf = mean_id(ratingsdf)
def ranking(uid, mid):
  (cmids, weights) = dictionary[mid] 
  rating = 0
  count = 0
  umr = ratingsdf.loc[ratingsdf['userId'] == uid]
  for i in range(len(cmids)):
    cm = cmids[i]
    w = weights[i]
    res = umr.loc[umr["movieId"] == cm].rating
    if not res.empty:
      rating += w * res.item()
      count += np.abs(w)
  return meandf.loc[meandf["userId"] == uid].rating_mean.item() if rating == 0 else rating / count
count = 0
total = 0

for index, row in ratingsdf.iterrows():
  if count > 20000:
    break
  uid = row["userId"]
  mid = row["movieId"]
  real_rank = row["rating"]
  pred_rank = meandf.loc[meandf["userId"] == uid].rating_mean.item()   
  count += 1
  total += ((pred_rank ) - real_rank) ** 2
  if (count % 200 == 0):
    print("we have proccessed " + str(count) + " rows")
t = np.sqrt(total / count)
print(t)

for index, row in ratingsdf.iterrows():
  if count > 20000:
    break
  uid = row["userId"]
  mid = row["movieId"]
  real_rank = row["rating"]
  pred_rank = ranking(uid, mid)
  count += 1
  total += ((pred_rank ) - real_rank) ** 2
  if (count % 200 == 0):
    print("we have proccessed " + str(count) + " rows")
t = np.sqrt(total / count)



print("holy fuck if this works")
print(t) 

