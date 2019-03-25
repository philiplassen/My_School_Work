#!/usr/bin/env bash

LINE="$(tail -n2 crawl_log.txt | head -n1 | rev | cut -d' ' -f1 | rev)"
while [ $LINE -lt 570 ]
do
  python3 crawl.py -v -n $LINE
  echo "going to sleep"
  sleep 1800
  LINE="$(tail -n2 crawl_log.txt | head -n1 | rev | cut -d' ' -f1 | rev)"
done 
