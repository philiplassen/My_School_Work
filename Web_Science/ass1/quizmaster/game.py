#!/usr/bin/env python3
from classifier import cleaned_data as data
print("Welcome to the Ultimate Quiz")
print("Please choose your topic, by selecting one of the following options")
print("(1) science-technology | (2) for-kids | (3) video-games | (4) sports | (5) music")
answer = input("Please write on of the numbers from above : ")
tmapping = {1 : "science-technology", 2 : "for-kids", 3 : "video games", 4 : "sports", 5 : "music"}
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
