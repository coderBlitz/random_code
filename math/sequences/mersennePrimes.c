#include<stdio.h>
#include<stdlib.h> 
#include<math.h>

int main(){
	int n=2;
	int MAX=200;
	int primes[MAX/2];
	int p=0;
	while(n<=MAX){//Makes an array of prime numbers
		int f=0;
		for(int j=1;j<=n;j++){
			if(n%j==0){f+=j;}
		}
		if(f==(n+1)){primes[p]=n;p++;}else{primes[p]=0;}
		n++;
	}
	for(int i=0;i<30;i++){printf("primes[%d]: %d\n",i,primes[i]);}
	n=0;
	while(n<(MAX)){//find mersenne primes
		unsigned int f=0;
		unsigned int k;
		if(primes[n]==0){n++;break;}
		else{k=pow(2,primes[n])-1;}
		for(unsigned int j=2;j<=k;j++){
			if((k%j)==0){f+=j;}
		}
		if(f==k){
			printf("%d\n",k);
		}
		n++;
	}
}
