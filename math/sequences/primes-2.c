#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(){
	unsigned long long max = 8e9;
	unsigned long count=0;
	char *primes = malloc(max * sizeof(char));
	if(primes == NULL){
		printf("Not enough memory!\n");
		exit(1);
	}
	for(unsigned long i=0;i<max;i++) primes[i] = 1;

	printf("GO!\n");
	for(unsigned long long i=2; i < max; i++){
		if(primes[i] == 0) continue;
		else{
//			printf("%lu\n",i);
			count++;
		}
		if(pow(i,2) < max){
			for(unsigned long n=i*2; n < max; n += i){
				primes[n] = 0;
			}
		}
	}
	printf("There are %d prime numbers up to %d\n",count,max);

	free(primes);
}
