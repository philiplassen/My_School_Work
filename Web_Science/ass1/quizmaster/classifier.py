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

result = data.groupby(by = "question")
print(result.groups)


