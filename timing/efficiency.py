import numpy as np
from time import time
from math import log2

def const(x):
	return x*x - 2*x + 1 # Help avoid minor optimization

def lin(x):
	total = 0
	for i in range(x):
		total += 2*i # Not just i, to hopefully avoid optimization out
	return total

def logn(x):
	s = x
	c = 0
	while s > 1.0:
		s /= 2
		c += 1
	return c


##########
# constant - N=16000 and a=4 is close to stdev below 0.1
# lin - N=500 and a=3 works, but is slow
# logn - N=4000 and a=3 gives consistent stdev below 0.1
M = 5
N = 500
a = 3
T = []
fn = logn

S = N
for k in range(M):
	print("Trial", k+1)
	start = time()

	for i in range(S):
		res = fn(i)

	stop = time()

	T.append(stop - start)
	S = N * a**(k+1)

print(T)

B = []
for k in range(M-1):
	B.append(T[k+1] / T[k])
print(B)

arr = np.array(B)
dev = np.std(arr)
mean = np.mean(arr)
print("Mean =", mean)
print("Std dev =", dev)
