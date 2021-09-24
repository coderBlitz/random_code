import math

primes = []
MAX = 1000000
for i in range(2,MAX):
	sum = 0
	for n in range(1,int(math.sqrt(i))+1):
		if i%n == 0:
			sum += n
	if sum == 1:
		primes.append(i)
	else:
		continue
print "There are %d prime numbers up to %d"%(len(primes),MAX)
