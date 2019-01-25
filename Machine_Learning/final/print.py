hyp = ["H" + str(i)  for i in range(1, 9)]
print(hyp)


data = [["d" + str(j + 1)] + [0 if i < j else 1 for i in range(8)] for j in range(8)]
for d in data:
  print(d)
