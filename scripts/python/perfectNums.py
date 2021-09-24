import time
import sys

n=2
MAX=10000000
start = time.clock()
avg = time.clock()

while n<=MAX:
	f=0
	for j in range(1,n):
		if n%j==0: f+=j
	
	if f==n:
		sys.stdout.write("\r%d is a Perfect Number                \n"%n)
		n += 2
		continue
	
	if (n%(MAX/200))==0:
		t = time.clock()
		sys.stdout.write("\r%ldm%lds %d%% through %d.  Average time per %d: %2lds"%((t-start)/60,(t-start)%60,n/(MAX/100),MAX,(MAX/200),t-avg))
		sys.stdout.flush()
		avg=time.clock()
		n += 2
		continue
	else:
		sys.stdout.write("\r%ldm%lds %d%% through %d."%((time.clock()-start)/60,(time.clock()-start)%60,n/(MAX/100),MAX))
		sys.stdout.flush()
	
	n+=2
print "\nTotal time was %ld:%ld"%((time.clock()-start)/60,(time.clock()-start)%60)
