#!/usr/bin/env python3
import sys
DEBUG = False
for arg in sys.argv:
  if arg == '-v':
    DEBUG = True
  
#Verbose for Debugging
def log(message):
  if DEBUG:
    log(message)

import collections

import pandas as pd
training_data = pd.read_csv("train_dataset.csv", header = None, encoding = "iso-8859-1", sep=";")

# Must transform into bag of data form.

training_data.columns = ["ID", "Question", "Answer", "Category"]

test_data =  pd.read_csv("question_answer.csv", header = None, names = ["Question", "Answer"], sep=";")

def cleanString(word):
  return "".join(s for s in word if (s.isalpha() or s.isdigit())).lower()  


word_sets = []
for index, row in test_data.iterrows():
  q = row["Question"]
  a = row["Answer"]
  qa = q.split() + a.split()
  qa = [cleanString(word) for word in qa]
  word_sets += [set(qa)]

def cleanString(word):
  return "".join(s for s in word if (s.isalpha() or s.isdigit())).lower()  

ranked = [0 for i in range(570)]


def cross_words(csv_file_path):
  df  = pd.read_csv(csv_file_path, header = None, names = ["Word"])
  for index, row in df.iterrows():
    w = row["Word"]
    for i in range(570):
      if w in word_sets[i]:
        ranked[i] = 1
  print(570 - sum(ranked))


cross_words("sport.csv")
cross_words("song.csv")
cross_words("games.csv")
cross_words("kids.csv")
cross_words("tech.csv")
"""
sports = pd.read_csv("sport.csv", header = None, names = ["Word"])
for index, row in sports.iterrows():
  w = row["Word"]
  for i in range(570):
    if w in word_sets[i]:
      ranked[i] = 1

log(570 - sum(ranked))

music  = pd.read_csv("song.csv", header = None, names = ["Word"])
for index, row in music.iterrows():
  w = row["Word"]
  for i in range(570):
    if w in word_sets[i]:
      if ranked[i] == 1:
        log("overlap at " + str(word_sets[i]))
      ranked[i] = 1

log(570 - sum(ranked))
"""
"""
for i in range(570):
  if ranked[i] == 0:
    print(word_sets[i])
"""


print(570 - sum(ranked))

lw = []
for index, row in test_data.iterrows():
  q = row["Question"]
  a = row["Answer"]
  qa = q.split() + a.split()
  qa = [cleanString(word) for word in qa]
  lw += qa

cnt = collections.Counter(lw)
print(cnt)
