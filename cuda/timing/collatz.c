#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<sys/time.h>

int main(){
//	uint64_t N = 1600000000;
	struct timeval start, end, diff;

	for(uint64_t N = 1e6;N < 100000000; N += 1000000){
//		printf("N = %lu\n", N);
		printf("%lu,", N);
		fflush(stdout);
		gettimeofday(&start, NULL);

		int *lens = malloc(N*sizeof(int));
		uint64_t max = 0;

		for(uint64_t i = 0;i < N;i++){
			int count = 0;
			uint64_t num = i;
			while(num > 1){
				if(num%2 == 0) num /= 2;
				else num = num*3 + 1;
				count++;
			}
			lens[i] = count;
			if(lens[i] > lens[max]) max = i;
		}
//		fprintf(stderr, "Largest chain: %d (%llu)\n", lens[max], max);

		free(lens);

		gettimeofday(&end, NULL);
		timersub(&end, &start, &diff);
//		printf("Total time: %2us %6uus\n\n\n", diff.tv_sec, diff.tv_usec);
		printf("%u.", diff.tv_sec);
		printf("%06u\n", diff.tv_usec);
		fflush(stdout);

	} // End N Loop

}
