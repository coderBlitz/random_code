p=2;
MAX=31;

while(p<=MAX):
	n = (2**p)-1;
	f=0;
	print "\rCurrent value of p: ",p;

	for x in range(1,n+1):
		if(n%x == 0): f += x;

	if f==(n+1):
		n *= 2**(p-1);
		print "\r%ld is a Perfect Number\n"%n;
	if (p==1 or p==2): p+=1
	else: p+=2;
