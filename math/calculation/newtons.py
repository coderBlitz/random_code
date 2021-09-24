def newton(f, F, xini, tol, maxIter):
	x = xini
	diff = tol + 1
	it = 0
	while it < maxIter and abs(diff) > tol:
		xn = x - f(x)/F(x)
		diff = xn - x
		x = xn
		it += 1
	if it == maxIter:
		print("Warning: Max iterations reached")

	return x

f = lambda x:x**2 - 2
F = lambda x:2*x

a = newton(f,F, 2, 0.000001, 30)
print(a)

f = lambda x: 100 + x*(40 - 2*x)
F = lambda x: 40 - 4*x
b = newton(f,F, 21, 0.000001, 30)
print(b)
