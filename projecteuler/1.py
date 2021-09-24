# Find the sum of all the multiples of 3 or 5 below 1000.

def s(n):
	return n*(n+1)//2

N = 1000 - 1
x = N // 3
y = N // 5
z = N // 15

total = 3*s(x) + 5*s(y) - 15*s(z)
print(total)
