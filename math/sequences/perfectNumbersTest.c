#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<signal.h>

/*By Chris Skane
Finds mersenne primes, then plugs then into equation for Perfect numbers and prints result
Due to data type limitations, max number of primes to be found is 8
*/

void handler(int sig){
	printf("\ninterrupted, exiting...\n");
	exit(0);
}

int main(){
	signal(SIGINT,handler);
	int p=2;
	int MAX=31;
	
	while(p<=MAX){
		unsigned long long n = pow(2,p)-1;
		unsigned long long f=0;
		printf("\rCurrent value of p: %d",p);
		fflush(stdout);

		for(long long j=1;j<=n;j++){
			if(n%j == 0){f += j;}
		}
		if(f==n+1){
			n *= pow(2,p-1);
			printf("\r%llu is a Perfect Number\n",n);
		}
		if(p==1 || p==2) p++;
		else p+=2;
	}
}
