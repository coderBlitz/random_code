# Find the difference between the sum of the squares of the first one hundred
#  natural numbers and the square of the sum

def ss(n):
	return n*(n+1)*(2*n+1)/6
def s(n):
	return n*(n+1)/2

N = 100

print(s(N)**2 - ss(N))
