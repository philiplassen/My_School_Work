#!/usr/bin/env python3

import sys
from classifier import cleaned_data as data
import random
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

print("Welcome to the ultimate Quiz")
print("")
print("We will ask you Questions of increasing difficulty")
print("You must pass all three levels to WIN THE GAME")

difficulty_map = { 1 : "Easy", 2 : "Medium", 3 : "Hard"}

level = 1
while (level < 4):
  print("You are on Level : " + difficulty_map[level])
  df = data.loc[data["difficulty"] == level]
  row = random.randint(0, df.shape[0] - 1)
  level += qa_row(df.iloc[row])


print("YOU HAVE COMPLETE THE GAME")

