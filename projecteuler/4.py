# Find the largest palindrome made from the product of two 3-digit numbers.

def palendrome(n):
	s = str(n)
	i = 0
	j = len(s)-1
	while i <= j:
		if s[i] != s[j]:
			return False
		i += 1
		j -= 1
	return True

m = 0
for i in range(100,999):
	for j in range(i,999):
		if palendrome(i*j):
			if i*j > m:
				m = i*j
print(m)

# Efficient way
# One of the two numbers *must* have a factor of 11
