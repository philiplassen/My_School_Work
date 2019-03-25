#!/usr/bin/env python3
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
scores = {"sports" : [], "music" : [], "video-games" : [], "for-kids" : [], "science-technology" : []}


def qa(question, answer):
  print(question)
  log(answer)
  guess = input("What is the answer? : ")
  return int(guess == answer)
    
def qa_row(series):
  log(series)
  return qa(series["question"], series["answer"])


cats = scores.keys()
for c in cats:
  df = data.loc[data["category"] == c]
  log(df)
  for i in range(2):
    row = random.randint(0, df.shape[0] - 1)
    scores[c] += [qa_row(df.iloc[row])]

log(scores)

 
