#!/usr/bin/env python3
import pandas as pd



data = pd.read_csv("crowd.tsv", encoding = "iso-8859-1", sep = '\t')

print(list(data.columns.values))
"""There are two components two this
First component is getting the majority vote of difficulty
we shall proceed with this below"""


"""I need to get the mean and mode of the factuality of the quesiton.
Below i try and join on The Question and than get the majority vote on
factuality and difficulty"""
print(data)

result = data.groupby(["question"])
print("before result")
print(result)
print("after result")
result = data.groupby(["question"])
count = 0

def category_to_int(cat):
  if cat == "Easy":
    return 1
  if cat == "Medium":
    return 2
  if cat == "Hard":
    return 3

data["difficulty"] = data["difficulty"].apply(category_to_int)


def majority_vote(votes, number_of_outcomes):
  a = [0] * number__of_outcomes
  for v in votes:
    a[v] += 1
  return a.index(max(a))


print(data["difficulty"])
for r in result:
  if count > 0:
    break
  print(r)
  print("after printing r")
  print(r[0])
  print("after first element has been printed")
  print(r[1]["difficulty"].values.tolist())
  print("after 2nd element has been printed")
  count += 1


