import random

N=10e6 # 10e6 = 10,000,000

num=random.randint(1,N)
print "Looking for number: %d..."%num

count=1
while 1:
	tmp = random.randint(1,N)
	if tmp == num:
		print "Found after %d numbers"%count
		print "Percent of all numbers guessed: %f"%(float(count/N))
		break
	count += 1