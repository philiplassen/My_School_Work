#!/usr/bin/env python3
import pandas as pd



data = pd.read_csv("crowd.tsv", encoding = "iso-8859-1", sep = '\t')

"""There are two components two this
First component is getting the majority vote of difficulty
we shall proceed with this below"""


"""I need to get the mean and mode of the factuality of the quesiton.
Below i try and join on The Question and than get the majority vote on
factuality and difficulty"""

result = data.groupby(["question"])
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
  a = [0] * (number_of_outcomes + 1)
  for v in votes:
    a[v] += 1
  return a.index(max(a))

test_data =  pd.read_csv("question_answer.csv", header = None, names = ["Question", "Answer"], sep=";")
test_labels = pd.read_csv("out.csv", header = None, names = ["Category"])



fdf = pd.DataFrame({"question" : [], "difficulty" : [], "opinion" : [], "factuality" : [], "answer": [], "category" : []})



for r in result:
  q = (r[0])
  idx_frame = (test_data[test_data["Question"] == q])
  if not idx_frame.empty:
    print(idx_frame)
    idx_frame = idx_frame.iloc[[0]]
    idx = idx_frame.index.item()
    print("index is " + str(idx))
    ans = test_data.iloc[[idx]]["Answer"].item()
    cat = test_labels.iloc[[idx]]["Category"].item()
    dif = majority_vote(r[1]["difficulty"].values.tolist(), 3)
    op = majority_vote(r[1]["opinion"].values.tolist(), 3)
    fac = majority_vote(r[1]["factuality"].values.tolist(), 2)
    temp = {"question" : q, "difficulty" : dif, "opinion" : op, "factuality" : fac, "answer" : ans, "category" : cat}
    fdf =fdf.append(temp, ignore_index = True)

cleaned_data = fdf

