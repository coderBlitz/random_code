#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<signal.h>

/*By Chris Skane
Finds mersenne primes, then plugs then into equation for Perfect numbers and prints result
Due to data type limitations, max number of primes to be found is 12
*/

void handler(int sig){
	printf("\ninterrupted, exiting...\n");
	exit(0);
}

double power(double b,int e){
	long p1=0,p2=0,base=b;
	if(e&1){
		p1 = (e-1)/2;
		p2 = (e-1)/2 + 1;
		for(int i=0;i<e;i++){
			b *= base;
		}
	}else{
		p1 = e/2;
		p2 = p1;
		b *= b;
		for(int i=0;i<=p1;i++){
			b *= base;
		}
	}
	return b;
}

int main(){
	signal(SIGINT,handler);
	int p=59;
	int MAX=61;
	double n = 0.0;
	
	while(p<=MAX){
		if(p > 31){n = (power(2,p)-1);}
		else{n = pow(2,p)-1;}
		double f=0.0;
		printf("\rCurrent value of p: %d\nCurrent value of n: %.0lf\n",p,n);
		fflush(stdout);

		for(double j=1.0;j<=n;j+=1.0){
			if((int)n%(int)j == 0.0){f += j;}
			if(f != 1 && f!= n+1){break;}
			/*if(j > 100000 && (j/(n/100) >= (double).5)){
				n *= pow(2,p-1);
				printf("\r%.0lf is a Perfect Number\n",n);
				break;
			}*/
			if((int)j%(int)n/100){
				printf("\r%.0f%% through %.0f. %lf",j/(n/100),n,j);
				fflush(stdout);
			}
		}
		if(f==n+1){
			n *= pow(2,p-1);
			printf("\r%.0lf is a Perfect Number\n",n);
		}
		if(p==1 || p==2) p+=1;
		else p+=2;
	}
printf("\n");
}
