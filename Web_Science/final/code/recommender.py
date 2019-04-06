#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity



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



"""Merge movies and ratings table"""


"""
Below are the lines to subtract the mean user rating from movie ratings.
This is done to get rid of the Rating bias introduced by the diifferent user.
"""
mean_user_rating = ratingsdf.groupby(['userId'], as_index = False, sort = False).mean().rename(columns = {'rating' : 'rating_mean'})[['userId', 'rating_mean']]
RatingsDF  = pd.merge(ratingsdf, mean_user_rating, on = 'userId', how = 'left', sort = False)
RatingsDF['adjusted'] = RatingsDF['rating'] - RatingsDF['rating_mean']
CFDF = pd.DataFrame({'userId' : RatingsDF['userId'], 'movieId' : RatingsDF['movieId'], "rating" : RatingsDF["adjusted"]})

"""
Loggin results of New DataFrame
"""
log("New CFDF  headers are : " + str(list(CFDF)))
log(CFDF)

result = CFDF.pivot_table(index = 'userId', columns = "movieId", values = "rating").fillna(0)
log(result)

""" What do I want to do????
For the whole dataset
-Calculate The Cosine Similarity

"""

def split_data(df):
    kf = KFold(n_splits=5, shuffle = True)
    errors = [pred_acc(df.ix[train_idx], df.ix[test_idx]) for train_idx, test_idx in kf.split(df)]
    print(errors)
     
def make_predictions(train_data):
  log("work in progress")
  print(train_data.shape)
  mean_user_rating = train_data.groupby(['userId'], as_index = False, sort = False).mean().rename(
  columns = {'rating' : 'rating_mean'})[['userId', 'rating_mean']]
  RatingsDF  = pd.merge(train_data, mean_user_rating, on = 'userId', how = 'left', sort = False)
  RatingsDF['adjusted'] = RatingsDF['rating'] - RatingsDF['rating_mean']
  CFDF = pd.DataFrame({'userId' : RatingsDF['userId'], 'movieId' : RatingsDF['movieId'], "rating" : RatingsDF["adjusted"]})
  result = CFDF.pivot_table(index = 'userId', columns = "movieId", values = "rating").fillna(0)
  correlation_matrix = similarity(result)
  print(correlation_matrix)
  weight_matrix = np.flip(np.sort(correlation_matrix), axis = 1)
  index_matrix = np.flip(np.argsort(correlation_matrix), axis = 1)
  user_ids = list(result.index)
  dictionary = {} 
  print(train_data.shape)
  for idx in range(correlation_matrix.shape[0]):
    p1  = [user_ids[i] for i in index_matrix[idx, 1:N+1]]
    p2 = weight_matrix[idx, 1:N + 1]
    key = user_ids[idx] 
    dictionary[key] = (p1, p2)
  return (CFDF, dictionary)
  """
  np.flip(sims[40, -(N + 1):610]))
  userIds = 
  """
def mean_id(data):
  mean_user_rating = train_data.groupby(['userId'], as_index = False, sort = False).mean().rename(
  columns = {'rating' : 'rating_mean'})[['userId', 'rating_mean']]
  return mean_user_rating
  
def similarity(pivot_table):
  return cosine_similarity(pivot_table)  


def get_ranking(movie_id, user_id, dic, CFDF):
  (indices, weights) = dic[user_id]
  total = 0
  rank = 0
  for i in range(len(indices)):
    rat = CFDF.loc[(CFDF["userId"] == [indices[i]]) & (CFDF["movieId"] == movie_id)]["rating"].values
    if len(rat) == 1:
      rank += rat[0] * weights[i]
      total += np.abs(weights[i])
  val = 0 if total == 0 else rank / total
  return val
  

def pred_acc(train_data, test_data):
  """
  Calculates the prediction error
  """
  print("getting prediction accuracy")
  print(train_data.shape)
  mean_user_rating = train_data.groupby(['userId'], as_index = False, sort = False).mean().rename(
  columns = {'rating' : 'rating_mean'})[['userId', 'rating_mean']]
  (CFDF, dictionary) = make_predictions(train_data)
  count = 0
  total = 0
  print("yo")
  print(train_data.shape)
  print(train_data.size)
  print(test_data.shape)
  for index, row in test_data.iterrows():
    uid = row["userId"]
    mid = row["movieId"]
    real_rank = row["rating"]
    pred_rank = get_ranking(mid, uid, dictionary, CFDF)
    mean = list(mean_user_rating.loc[mean_user_rating["userId"] == uid]["rating_mean"])[0]
    count += 1
    total += ((pred_rank + mean) - real_rank) ** 2
    if (count % 1000 == 0):
      print("we have proccessed " + str(count) + " rows")
  t = np.sqrt(total / count)
  print(t)
  return(t)

  """ For each value in test_data 
  Get the mean rating for the user
  Get their rating
  subrtract the two
  call get_ranking()
  calculate the difference"""

make_predictions(ratingsdf)
print("starting the task")
val = split_data(ratingsdf)
print(val)

"""
mergedDF = pd.merge(, RatingsDF, on = 'movieId', how = 'left', sort = False)
result = mergedDF.pivot_table(index = 'userId', columns = 'movieId', values = 'rating')
result1 = mergedDF.pivot_table(index = 'userId', columns = 'movieId', aggfunc = 'mean')
log("Merged headers are : " + str(list(mergedDF)))
log(result)
log(result1)
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

