# By considering the terms in the Fibonacci sequence whose values do not exceed
#  four million, find the sum of the even-valued terms.

N = 4000000
a = 1
b = 2

total = 2
while b <= N:
	c = a + b

	a = b
	b = c

	if c % 2 == 0:
		total += c
print(total)

# Efficient method (after solved)
a = 0
b = 2
c = 0

total = 2
while c <= N:
	total += c

	c = 4*b + a

	a = b
	b = c
print(total)
