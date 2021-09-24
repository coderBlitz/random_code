# What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?

# Product of largest power of all primes, that is under 20, for 1-20
from math import sqrt,gcd,log,floor

def isprime(n):
	for i in range(2,int(sqrt(n))+1):
		if gcd(n,i) != 1:
			return False
	return True

N = 20

product = 1
for i in range(2,N+1):
	# Only check primes
	if not isprime(i):
		continue

	# Check highest power of prime that is <= N
	j = floor(log(N)/log(i))
	print(i, j)
	product *= i**j
print(product)

# Efficient method
# Modify above to use floor of logarithms. log(N)/log(i)
