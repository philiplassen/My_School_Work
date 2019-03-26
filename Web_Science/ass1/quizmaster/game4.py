#!/usr/bin/env python3
import operator
import pandas as pd
import sys
import random
from classifier import cleaned_data as data
DEBUG = False
if '-v' in sys.argv:
  DEBUG = True

def log(message):
  if DEBUG:
    print(message)

def qa(question, answer):
  print(question)
  log(answer)
  guess = input("What is the answer? : ")
  return int(guess == answer)
    
def qa_row(series):
  return qa(series["question"], series["answer"])

user = int(input("What is your user ID? : "))
test_data =  pd.read_csv("question_answer.csv", header = None, names = ["Question", "Answer"], sep=";")
test_labels = pd.read_csv("out.csv", header = None, names = ["Category"])


crowd_data = pd.read_csv("crowd.tsv", encoding = "iso-8859-1", sep = '\t')
users_data = crowd_data.loc[crowd_data["id"] == user]
topics = ["science-technology", "for-kids",  "video-games",  "sports",  "music"]
topic_scores = {t: [] for t in topics}
for q in users_data["question"]:
  op = users_data.loc[(users_data["question"] == q) & (users_data["id"] == user)]['opinion'].values[0]
  print(op)
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

def qa(question, answer):
  print(question)
  log(answer)
  guess = input("What is the answer? : ")
  return int(guess == answer)
    
def qa_row(series):
  return qa(series["question"], series["answer"])


df_cat = data.loc[data["category"] == fav_cat]
while (True):
  row = random.randint(0, df_cat.shape[0] - 1)
  qa_row(df_cat.iloc[row])
