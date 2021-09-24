# What is the index of the first term in the Fibonacci sequence to contain 1000 digits?

from math import log10,sqrt,floor,ceil

# Credit to geeksforgeeks for idea.
# Using Binet's formula (modified), find digits

PHI = (1 + sqrt(5))/2.0

def digits(n):
	return ceil(n*log10(PHI) - log10(5)/2)

N = 999

n = ceil((N + log10(5)/2)/log10(PHI))
print("F(", n, ") has", N, "digits")

print(digits(n))
