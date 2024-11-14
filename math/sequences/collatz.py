lens=[]
MAX=0
high=1024*4096

for N in range(high):
	if N%(high/10) == 0:
		print "%d percent"%((N/(high/10)) * 10)
	count=0
	num = N
	while num>1:
		if (num%2) == 0:
			num /= 2
		else:
			num = num*3 + 1
		count += 1
	lens.append(count)
	if lens[N] > lens[MAX]:
		MAX = N

print "%lu has the longest chain"%MAX
print "Chain length: %d"%lens[MAX]
