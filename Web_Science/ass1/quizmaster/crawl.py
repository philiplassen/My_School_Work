#!/usr/bin/env/ python3
import sys
from googlesearch import search
import pandas as pd

DEBUG = False
if '-v' in sys.argv:
  DEBUG = True

if DEBUG:
  print("Using Debug mode = Verbose printing")

def log(message):
  if DEBUG:
    print(message)


query = "Who released the album fearless"
result = search(query, num = 40, pause = 9.0, stop = 40)
count = 0
music_words = {"song", "single", "album", "artist", "singer", "music"}
sport_words = {"sport", "olympics", "medal", "athlete", "competition", "win"}
game_words = {"xbox", "playstation", "games", "nintendo", "wii"}
kids_words = {"kid", "series"}
technology_words = {"science", "chem", "physic", "math", "computer"}
word_map = {"sports" : sport_words, "music" : music_words, "game_words" : game_words, "for-kids" : kids_words, "science-technology" : technology_words}
count_map = {topic : 0 for topic in word_map.keys()}
for r in result:
  for (topic, words) in word_map.items():
    count = 0
    for w in words:
      if w in r.lower():
        count += 1
    count_map[topic] += count

print(count_map)


