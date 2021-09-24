import random

def myRand(x):
	random.seed(x)
	return random.randrange(0, 2**32)


random.seed()
N = 30
tests = [random.randrange(0, 2**32) for i in range(N)]

# Lambda = 58942 for windows testing
for j in tests:
	print("j =", j)
	# Cycle loop
	lam = 1
	p = 1
	fn = myRand
	t = j # x0
	h = fn(t)
	while t != h:
		if lam == p:
			t = h
			p *= 2
			lam = 0

		h = fn(h)
		lam += 1

	print("Lambda =", lam)

	# Move lambda to start of cycle
	t = j
	h = t
	for i in range(lam):
		h = fn(h)

	# Find length of cycle
	mu = 0
	while t != h:
		t = fn(t)
		h = fn(h)

		mu += 1

	print("Mu =", mu)
