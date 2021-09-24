# Find the value of d < 1000 for which 1/d contains the longest recurring cycle in its decimal fraction part.

# Use floyd's cycle detection on the decimal places (if double-precision has enough digits)

import sys

#i = 7

N = 1000
lens = []
for i in range(1,N+1):
	val = 1/i

	# Check if non-repeating
	skip = False
	prod = val
	for _ in range(14):
		prod *= 10
		prod -= int(prod)
		if prod == 0:
			#print(i, "Non-repeating")
			skip = True
			break
	if skip:
		continue

	a = 10 * val
	b = 100 * val
	j = 1
	while int(a % 10) != int(b % 10) and j < 10:
		a *= 10
		b *= 100
		j += 1

	if int(a % 10) != int(b % 10):
		continue

	if j > 6:
		print("1 /", i, "has cycle length", j)
	lens.append(j)

res = max(lens)
print("Max:", res)
for i,num in enumerate(lens,1):
	if num == res:
		print(i)
