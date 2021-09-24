from math import ceil, cos, isclose, pi

def inverse(num):
	return 1/num

def factor(X, Y):
	if len(X) == 1:
		return Y[0]
	else:
		A = factor(X[:-1], Y[:-1])
		B = factor(X[1:], Y[1:])
		res = (B - A)/(X[-1] - X[0])
		#print("Returning:", res)
		return res

# Given X and Y values, print an equation estimating the function Y=f(X)
def print_estimate(X, Y):
	a = []
	for i in range(1, len(X)+1):
		a.append(factor(X[:i], Y))

	print(Y[0], end="")
	for i in range(1, len(X)):
		temp = X[:i]
		if a[i] < 0:
			sign = "-"
		else:
			sign = "+"

		divide = False
		inverted = abs(1/a[i])
		if inverted % 1 == 0:
			# If integer divisor
			divide = True

		if not divide:
			print("",sign, "{:.10f}".format(abs(a[i])), end="")
		else:
			print("",sign,end=" ")
		for x in temp:
			b = ceil(x)
			if isclose(x, b):
				print("(x-{:d})".format(b), end="")
			else:
				print("(x-{:.4f})".format(x), end="")
		if divide:
			print("/", int(inverted), sep="", end="")
	print()

	return a

# Just a simple test with a known function
X = [1, 2, 8, 10]
Y = [inverse(x) for x in X]

a = print_estimate(X, Y)
print("Factor:", a)

X2 = [0, pi/6, pi/4, pi, 2*pi]
Y2 = [cos(x) for x in X2]

b = print_estimate(X2, Y2)
print("Factor:", b)

def poly(num):
	return num*num

X3 = [2, 3, 4, 7]
Y3 = [poly(x) for x in X3]
b = print_estimate(X3, Y3)
print("Factor:", b)
