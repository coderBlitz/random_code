// Find the sum of all the primes below two million.

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(){
	unsigned long max = 2e6;
	unsigned long count=0;
	char *primes = malloc(max * sizeof(char));
	for(unsigned long i=0;i<max;i++) primes[i] = 1;

	for(unsigned long long i=2; i < max; i++){
		if(primes[i] == 0) continue;
		if(pow(i,2) < max){
			for(unsigned long n=i*2; n < max; n += i){
				primes[n] = 0;
			}
		}
	}

	unsigned long sum = 0;
	for(unsigned long i=2;i < max;++i) if(primes[i]) sum += i;

	printf("%ld\n",sum);
	free(primes);
}
