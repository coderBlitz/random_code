#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<time.h>

int main(){
	unsigned long long max = 1e8;
	unsigned long count=0;

	FILE *plot = popen("gnuplot -p","w");
	fprintf(plot, "set title 'C primes'\n");
	fprintf(plot, "set terminal qt size 1280,720\n");

	char *primes = malloc(max * sizeof(char));
	for(unsigned long i=0;i<max;i++) primes[i] = 1;

	fprintf(plot,"plot '-' every 5 pt 7\n");
	printf("GO!\n");
	for(unsigned long long i=2; i < max; i++){
		if(primes[i] == 0) continue;
		else{
//			printf("%lu\n",i);
			fprintf(plot, "%lu %lu\n", count, i);
			count++;
		}
		if(pow(i,2) < max){
			for(unsigned long n=i*2; n < max; n += i){
				primes[n] = 0;
			}
		}
	}
	fprintf(plot, "e");

	fclose(plot);
}

