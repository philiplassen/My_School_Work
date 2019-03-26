#!/usr/bin/env python3
import operator
import numpy as np
import random
import sys
from classifier import cleaned_data as data

DEBUG = False
if '-v' in sys.argv:
  DEBUG = True

def log(message):
  if DEBUG:
    print(message)

print("Welcome to the Ultimate Quiz")

# Dictionary that holds the users scores for question from each topic
initial_scores = [1, 1]
scores= {"sports" : [1, 1], "music" : [1, 1], "video-games" : [1, 1], "for-kids" : [1, 1], "science-technology" : [1, 1]}


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




def qa(question, answer):
  print(question)
  log(answer)
  guess = input("What is the answer? : ")
  return int(guess == answer)
    
def qa_row(series):
  return qa(series["question"], series["answer"])

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
    

while (not is_double(scores) and n < 25): 
  n += 1
  c = choose_cat(scores) 
  log(c)
  df = data.loc[data["category"] == c]
  row = random.randint(0, df.shape[0] - 1)
  print(scores[c])
  scores[c] += [qa_row(df.iloc[row])]
  log(scores)

def get_max_key(scores):
  return max(scores.items(), key=operator.itemgetter(1))[0]  

cat = get_max_key(scores)
df_cat = data.loc[data["category"] == cat]
while (True):
  row = random.randint(0, df_cat.shape[0] - 1)
  qa_row(df_cat.iloc[row])
