def factorial(n):
	total=1
	for x in range(n):
		total *= (n-x)
	return total

total=pieces=possibilities=0
pieces = input("Number of Possibilities\nHow many objects are there: ")

total = input("How many total spaces are there including ones with pieces on them: ")

possibilities = factorial(total)/(factorial(total-3)*factorial(pieces));
print "There are %d unique possibilities"%possibilities
