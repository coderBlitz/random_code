# Lattice paths
#	Starting in the top left corner of a square grid, and only moving right and
#	 down, there are 6 routes to the bottom-right corner of a 2x2 grid.
#
#	How many routes are there through a 20x20 grid?


# Notes: Sum of squares
#
# Implicit:
#	lattice(n, m) = lattice(n, m-1) + lattice(n-1, m)

# 1x1 ==> 2
# 2x2 ==> 6
# 3x3 ==> 20

res = [[0] * 25] * 25
def lat(n, m):
	if n <= 0 or m <= 0:
		return 0
	if n == 1 or m == 1:
		return n + m

	#print("(",n,",",m,")")
	# Check for existing result
	global res
	if res[n][m] != 0:
		return res[n][m]

	ret = lat(n-1, m) + lat(n, m-1)
	#print("(",n,",",m,") =>", ret)
	#res[n][m] = ret
	return ret

#print(res)
end = 20
#for n in range(end+1):
	#print(n, "==>" , lat(n, n))


# Solution (inspired by solution type, e.g combinatorics, not explicit answer):
# lattice(n) = (2n choose n)
def fact(n):
	if n == 1 or n == 0:
		return 1
	return n * fact(n-1)

def choose(n, m):
	if n < 0 or m < 0 or m > n:
		raise ValueError("M must be leq N, and both must be strictly positive")
	return fact(n) // (fact(n - m) * fact(m))

for i in range(1, end+1):
	print(i, "==>" , choose(2*i, i))
