import sys
DEBUG = False
for arg in sys.argv:
  if arg == '-v':
    DEBUG = True
  
#Verbose for Debugging
def log(message):
  if DEBUG:
    print(message)

import pandas as pd
df = pd.read_csv("train_dataset.csv", header = None, encoding = "iso-8859-1", sep=";")
log(list(df.columns.values))

# Must transform into bag of data form.

df.columns = ["ID", "Question", "Answer", "Category"]



