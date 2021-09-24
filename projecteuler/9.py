# There exists exactly one Pythagorean triplet for which a + b + c = 1000.
# Find the product abc.

# Notes:
# since a < b < c, 335 <= c <= 997 otherwise a+b+c < 1000 or a = b or b = c

# Start at 997, and smallest 'a', then calculate 'b' from 'a' and repeat

from math import sqrt,ceil,gcd

A = B = C = 0

for c in range(997,335,-1):
	for a in range(1,333):
		b = sqrt(c**2 - a**2)

		S = a+b+c
		if S == 1000:
			A = a
			B = b
			C = c
			print("a,b,c =", a,b,c)
			break

print("abc =", A*B*C)

# Efficient method (parameterization)
S = 1000
s2 = S/2
mlim = ceil(sqrt(s2)) - 1

for m in range(2,mlim):
	if s2 % m == 0:
		sm = s2 / m
		while sm % 2 == 0:
			sm /= 2

		if m % 2 == 1:
			k = m + 2
		else:
			k = m + 1

		while k < 2*m and k <= sm:
			if sm % k == 0 and gcd(k,m) == 1:
				d = s2 / (k*m)
				n = k - m
				a = d*(m*m - n*n)
				b = 2*d*m*n
				c = d*(m*m + n*n)
				print(a,b,c)
			k += 2
