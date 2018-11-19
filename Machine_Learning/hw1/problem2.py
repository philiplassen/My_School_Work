import random
import matplotlib.pyplot as pl



#Generate a list of a million experiments
experiments = [[random.randint(0, 1) for i in range(20)] for i in range(1000000)]
print("Experiments have been generated")

#calculate the averages
sample_averages = [float(sum(x))/20 for x in experiments]
print("Sample averages have been calculated")

#delete this when done
print(sample_averages[8:11])
print()

alphas = [.5 + i * 0.05 for i in range(11)]


frequencies = [0 for i in range(0, 11)]

for val in sample_averages:
  for j in range(11):
    if val >= alphas[j]:
      frequencies[j] += float(1)/1000000

#plotting alphas against frequencies


#Markov's inequality is "P(X >= a) <= E(X)/a" for reference.
markov_vals = [.5 / alpha for alpha in alphas]
print(markov_vals)

cheb_vals = [ .25 / (alpha * alpha) for alpha in alphas] 
print(cheb_vals)

pl.xlabel("Alpha Value")
pl.ylabel("Result")

pl.plot(alphas, frequencies, marker='o', label='Line 1')
pl.plot(alphas, markov_vals, marker='o', label='Line 2')
pl.plot(alphas, cheb_vals, marker='o', label='Line 3')

pl.legend(["Emperical Frequencies", "Markov's Bound", "Chebyshev's Bound"])

pl.show()


print(frequencies)


