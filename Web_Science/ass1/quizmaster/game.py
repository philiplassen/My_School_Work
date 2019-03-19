#!/usr/bin/env python3

print("Welcome to the Ultimate Quiz")
print("Please choose your topic, by selecting one of the following options")
print("(1) science-technology | (2) for-kids | (3) video-games | (4) sports | (5) music")
answer = input("Please write on of the numbers from above : ")
mapping = {1 : "science-technology", 2 : "for-kids", 3 : "video games", 4 : "sports", 5 : "music"}
try:
  val = int(answer)
  print("you selected the topic : " + mapping[val])
except ValueError:
  print("you did not input an integer")

  
