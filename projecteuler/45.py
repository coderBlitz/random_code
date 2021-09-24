# It can be verified that T_285 = P_165 = H_143 = 40755.

# Find the next triangle number that is also pentagonal and hexagonal.


# "Walk" each number until then match. Then walk again. (or start higher)

def Tn(n):
	return n * (n+1) // 2
def Pn(n):
	return n * (3*n - 1) // 2
def Hn(n):
	return n * (2*n - 1)


t = 2
p = 2
h = 2

tick = False
while True:
	tn = Tn(t)
	pn = Pn(p)
	hn = Hn(h)

	if tn < pn or tn < hn:
		t += 1
	if pn < hn or pn < tn:
		p += 1
	if hn < tn or hn < pn:
		h += 1

	if tn == pn == hn:
		if tick:
			break
		tick = True # Looking for second instance, so stop on second
		t += 1 # Increment to get the loop going again
print(tn)
