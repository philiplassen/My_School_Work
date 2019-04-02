#!/usr/bin/env python3
import pandas as pd
import sys

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
mergedDF = pd.merge(moviesdf, ratingsdf, on = 'movieId', how = 'left', sort = False)


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

