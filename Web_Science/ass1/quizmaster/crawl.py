#!/usr/bin/env python3
import operator
import sys
import random
import csv
import time
import pandas as pd
from googlesearch import search

DEBUG = False
if '-v' in sys.argv:
  DEBUG = True
index = 0
if '-n' in sys.argv:
  index = int(sys.argv[len(sys.argv) - 1]) -1
  
if DEBUG:
  print("Using Debug mode = Verbose printing")

def log(message):
  if DEBUG:
    print(message)


#result = search(query, num = 40, pause = 9.0, stop = 40)
count = 0
music_words = {"song", "single", "album", "artist", "singer", "music"}
sport_words = {"sport", "olympics", "medal", "athlete", "competition", "win"}
game_words = {"xbox", "playstation", "game", "nintendo", "wii"}
kids_words = {"kid", "series", "history", "child", "happy", "cartoon"}
technology_words = {"science", "chem", "physic", "math", "computer"}
word_map = {"sports" : sport_words, "music" : music_words, "video-games" : game_words, "for-kids" : kids_words, "science-technology" : technology_words}
count_map = {topic : 0 for topic in word_map.keys()}
test_data =  pd.read_csv("question_answer.csv", header = None, names = ["Question", "Answer"], sep=";")
wtr = csv.writer(open ('out.csv', 'a'), delimiter=',', lineterminator='\n')
q_count = index
time_start = time.time()
for q in test_data["Question"][index:]:
  q_count += 1
  log("Question Number : " + str(q_count))
  print("Question Number: " + str(q_count), file = open("crawl_log.txt", "a"))
  log(q)
  print(q, file = open("crawl_log.txt", "a"))
  count_map = {topic : 0 for topic in word_map.keys()}
  url_num = random.randint(15, 41)
  wait_time = round(random.randrange(0, 1), 2) + random.randint(1, 4)
  result = search(q, num = url_num, pause = wait_time, stop = url_num)
  for r in result:
    for (topic, words) in word_map.items():
      count = 0
      for w in words:
        if w in r.lower():
          count += 1
      count_map[topic] += count
  label = max(count_map.items(), key=operator.itemgetter(1))[0]
  log(label) 
  print(label , file = open("crawl_log.txt", "a"))
  log(count_map)
  print(count_map, file = open("crawl_log.txt", "a"))
  wtr.writerow([label])  
time_stop = time.time()
log(time_start - time_stop)
log("The time to classify " + str(q_count) + " Questions is " + str(time_start - time_stop))



