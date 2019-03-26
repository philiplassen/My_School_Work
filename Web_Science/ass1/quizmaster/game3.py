#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
#from classifier import cleaned_data as data
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
DEBUG = False
if '-v' in sys.argv:
  DEBUG = True

def log(message):
  if DEBUG:
    print(message)


"""Need to figure out what the best format of the data is"""

df = pd.read_csv("crowd.tsv", encoding = "iso-8859-1", sep = '\t')
df = df[['id', 'question', 'opinion']]
df


# In[15]:


mean = df.groupby(['id'], as_index = False, sort = False).mean().rename(columns = {'opinion' : 'opinion_mean'})[['id', 'opinion_mean']]
Opinions = pd.merge(df, mean, on = 'id', how = 'left', sort = False)
Opinions['adjusted'] = Opinions['opinion'] - Opinions['opinion_mean']
Opinions


# In[19]:


result = pd.DataFrame({'id':Opinions['id'], "question": Opinions['question'], "opinion" : Opinions['adjusted']})
result1 = result.pivot_table(index = 'id', columns = 'question', values = 'opinion').fillna(0)
result1


# In[47]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(result1)
resulty = np.argsort(cosine_similarity(result1))
user = 10
k = 4
np.flip(resulty[10, -(k+1): 37])


# In[26]:





# In[ ]:




