# What is the largest prime factor of the number 600851475143 ?
from math import sqrt,gcd

def isprime(n):
	for i in range(2,int(sqrt(n))):
		if gcd(n,i) != 1:
			return False
	return True

N = 600851475143

a = int(sqrt(N))
if a % 2 == 0:
	a -= 1

while a > 1:
	if N % a == 0:
		if isprime(a):
			break
	a -= 2
print(a)
