import random
import matplotlib.pyplot as pl


x = [10,
20,
40,
80,
100,
130,
160]

y = [
1682910,
4635845,
11291060,
32460847,
43200000,
63409600,
82235834]

y1 = [
42482041,
143069340,
359358329,
664329770,
1030000000,
1237547513,
1518166907]


z = [
5.77E-06,
4.33E-06,
3.55E-06,
2.47E-06,
2.32E-06,
2.05E-06,
1.95E-06]


z1 = [
2.37E-07,
1.40E-07,
9.54E-08,
8.24E-08,
7.78E-08,
6.95E-08,
6.33E-08]

pl.xlabel("Threads")
pl.ylabel("Throughput")
pl.title("Throughput Plot")
pl.plot(x, z, marker='o', label='Emperical Frequeuncies')
pl.plot(x, z1, marker='o', label='Markov Bounds')

pl.legend(["Same Adress Space", "Different Adress Space"])
pl.show()
