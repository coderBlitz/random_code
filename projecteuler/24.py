# A permutation is an ordered arrangement of objects. For example, 3124 is one
#  possible permutation of the digits 1, 2, 3 and 4. If all of the permutations
#  are listed numerically or alphabetically, we call it lexicographic order.
# The lexicographic permutations of 0, 1 and 2 are:
#	012   021   102   120   201   210
# What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?


# Add factorials while less than 1 million, using division to find digit

def fact(n):
	if n == 1 or n == 0:
		return 1
	return n * fact(n-1)

N = 1000001

total = N
digits = "12345679"
for i in range(10):
	print(str(i) in digits)
