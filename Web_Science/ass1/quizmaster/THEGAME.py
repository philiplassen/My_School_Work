#!/usr/bin/env python3
import operator
import pandas as pd
import numpy as np
from classifier import cleaned_data as data
import random
import sys
DEBUG = False
if '-v' in sys.argv:
  DEBUG = True

Sim = False
player_profile = 0
topics = ["science-technology", "for-kids",  "video-games",  "sports",  "music"]
if '-s' in sys.argv:
  Sim = True
  player_profile = {t : float(input("Accuracy for " + t + " ? : ")) for t in topics}
  
if '-s1' in sys.argv:
  Sim = True
  player_profile = {"science-technology" : .2, "for-kids" : .2,  "video-games" : .2,  "sports" : .2,  "music" : .8}
  

if '-s2' in sys.argv:
  Sim = True
  player_profile = {"science-technology" : .2, "for-kids" : .2,  "video-games" : .2,  "sports" : .2,  "music" : .333}
 

def log(message):
  if DEBUG:
    print(message)


def keep_going():
  if Sim:
    return True
  return input("Would you like to keep Guessing | 0 = No | 1 = Yes ? : ") == "1"


def qa(question, answer):
  print(question)
  log(answer)
  guess = input("What is the answer? : ")
  if (guess == answer):
    print("Correct!")
  else:
    print("False")
  return int(guess == answer)
    
def qa_row(series):
  if Sim:
    categ = series["category"]
    pr = player_profile[categ]
    return np.random.uniform() < pr
  return qa(series["question"], series["answer"])
  
if '-1' in sys.argv:
  print("Welcome to the Ultimate Quiz")
  print("Please choose your topic, by selecting one of the following options")
  print("(1) science-technology | (2) for-kids | (3) video-games | (4) sports | (5) music")
  answer = input("Please write on of the numbers from above : ")
  tmapping = {1 : "science-technology", 2 : "for-kids", 3 : "video-games", 4 : "sports", 5 : "music"}


  try:
    topic = int(answer)
    print("you selected the topic : " + tmapping[topic])
  except Exception:
    print("you did not input a valid integer")


  print("Please select your desired difficulty level")
  print("(1) Easy | (2) Medium | (3) Hard")
  difficulty = input("Please select one of the numbers above : ")
  dmapping = {1 : "easy", 2 : "medium", 3 : "hard"}
  try:
    difficulty = int(difficulty)
    print("you selected the difficulty : " + dmapping[difficulty])
  except Exception:
    print("you did not input a valid Integer")
  df = data.loc[(data["category"] == tmapping[topic]) & (data["difficulty"] == difficulty) & (data["factuality"] == 0)]
  print(df)
  Go = True
  while(Go):
    row = random.randint(0, df.shape[0] - 1)
    qa_row(df.iloc[row])
    Go = keep_going()

if '-2' in sys.argv:

  # Dictionary that holds the users scores for question from each topic
  scores= {"sports" : [1, 0, 0], "music" : [1, 0, 0] ,"video-games" : [1, 0, 0], "for-kids" : [1, 0, 0], "science-technology" : [1, 0, 0]}



  def choose_cat(scores):
    accuracy = {c : sum(scores[c])/ len(scores[c]) for c in scores.keys()}
    print(accuracy)
    total = sum(list(accuracy.values()))
    tot = 0
    probs = {c : accuracy[c] / total for c in scores.keys()}
    thresh = {}
    for key, value in probs.items():
      tot += value 
      thresh[key] = tot
    r = np.random.uniform()
    print(thresh.items())
    for key, value in thresh.items():
      if r < value:
        return key
        
  n = 0 #number of total questions asked
  def is_double(scores):
    for k in scores.keys():
      is_double = True
      for j in scores.keys():
        if scores[k] < 2 * scores[j] and k != j:
          is_double = False
      if is_double:
        return True
    return False
  old_acc = {c : 100 for c in scores.keys()} 
  accuracy = {c : sum(scores[c])/ len(scores[c]) for c in scores.keys()} 
  dif = sum([np.abs(old_acc[c] - accuracy[c]) for c in scores.keys()])
  print(old_acc)
  print(accuracy)
  print(dif)
  c = choose_cat(scores)
  while ( dif < 0.00001  or dif  > .002 and n < 1000): 
    n += 1
    c = choose_cat(scores) 
    log(c)
    old_acc = accuracy
    df = data.loc[data["category"] == c]
    row = random.randint(0, df.shape[0] - 1)
    scores[c] += [qa_row(df.iloc[row])]
    accuracy = {c : sum(scores[c])/ len(scores[c]) for c in scores.keys()} 
    dif = sum([np.abs(old_acc[c] - accuracy[c]) for c in scores.keys()])
    print(old_acc)
    print(accuracy)
    print(dif)
    log(scores)

  print("number of questions before convergence is : " + str(n))
  def get_max_key(scores):
    return max(scores.items(), key=operator.itemgetter(1))[0]  

  cat = get_max_key(accuracy)
  print(cat)
  df_cat = data.loc[data["category"] == cat]
  while (not Sim):
    row = random.randint(0, df_cat.shape[0] - 1)
    qa_row(df_cat.iloc[row])
    keep_going()

if '-3' in sys.argv:

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


if '-4' in sys.argv:
  test_data =  pd.read_csv("question_answer.csv", header = None, names = ["Question", "Answer"], sep=";")
  test_labels = pd.read_csv("out.csv", header = None, names = ["Category"])

  user = int(input("What is your ID? : "))

  crowd_data = pd.read_csv("crowd.tsv", encoding = "iso-8859-1", sep = '\t')
  users_data = crowd_data.loc[crowd_data["id"] == user]
  topics = ["science-technology", "for-kids",  "video-games",  "sports",  "music"]
  topic_scores = {t: [] for t in topics}
  for q in users_data["question"]:
    op = users_data.loc[(users_data["question"] == q) & (users_data["id"] == user)]['opinion'].values[0]
    idx_frame = (test_data[test_data["Question"] == q])
    if not idx_frame.empty:
      idx_frame = idx_frame.iloc[[0]]
      idx = idx_frame.index.item()
      cat = test_labels.iloc[[idx]]["Category"].item()
      topic_scores[cat] += [op]

  def get_max_key(scores):
    return max(scores.items(), key=operator.itemgetter(1))[0]  

      
  def mean(array):
    return sum(array) / len(array) if len(array) > 0 else 0

  topic_means = {t: mean(topic_scores[t]) for t in topics}
  fav_cat = get_max_key(topic_means)
  log(fav_cat)
  log(topic_scores)
  log(topic_means)

  df_cat = data.loc[data["category"] == fav_cat]
  Go = True 
  while (Go):
    row = random.randint(0, df_cat.shape[0] - 1)
    qa_row(df_cat.iloc[row])
    Go = input("Would you like to keep Guessing | 0 = No | 1 = Yes ? : ") == "1"

if '-5' in sys.argv:
  print("Welcome to the ultimate Quiz Levels")
  print("")
  print("We will ask you Questions of increasing difficulty")
  print("You must pass all three levels to WIN THE GAME")

  difficulty_map = { 1 : "Easy", 2 : "Medium", 3 : "Hard"}

  level = 1
  Go = True 
  while (level < 4 and Go):
    print("You are on Level : " + difficulty_map[level])
    df = data.loc[data["difficulty"] == level]
    row = random.randint(0, df.shape[0] - 1)
    level += qa_row(df.iloc[row])
    Go = keep_going()


  print("YOU HAVE COMPLETE THE GAME")

