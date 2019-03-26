#/usr/bin/env python3

import sys
from classifier import cleaned_data as data
import random
DEBUG = False
if '-v' in sys.argv:
  DEBUG = True

def log(message):
  if DEBUG:
    print(message)


