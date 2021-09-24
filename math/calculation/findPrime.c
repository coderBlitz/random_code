#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(){
	unsigned long max = 1e9;
	unsigned long count=0, prime = 0;
	unsigned int N = 0;
	char *primes = malloc(max * sizeof(char));
	for(unsigned long i=0;i<max;i++) primes[i] = 1;

	printf("Which prime would you like to find: ");
	scanf("%u", &N);

	printf("GO!\n");
	for(unsigned long long i=2; i < max; i++){
		if(primes[i] == 0) continue;
		else{
//			printf("%lu\n",i);
			count++;
			if(count == N){
				printf("Found it\n");
				prime = i;
				break;
			}
		}
		if(pow(i,2) < max){
			for(unsigned long n=i*2; n < max; n += i){
				primes[n] = 0;
			}
		}
	}

	printf("%ld\n",prime);
	free(primes);
}
