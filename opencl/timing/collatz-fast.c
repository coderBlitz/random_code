#include<stdio.h>
#include<stdlib.h>
#include<time.h>

void timespecsub(struct timespec *a, struct timespec *b, struct timespec *res){
	res->tv_sec = a->tv_sec - b->tv_sec;
	res->tv_nsec = a->tv_nsec - b->tv_nsec;
	if(res->tv_nsec < 0){
		res->tv_nsec += 1000000000;
		res->tv_sec -= 1;
	}
}

int main(){
	unsigned long long N_max = 0x40 << 22;
	unsigned int *lens = malloc(N_max*sizeof(*lens));
	unsigned long max = 0;

	struct timespec start, end, diff;

	for(unsigned long N = 0x40;N <= N_max;N <<= 1){
		//printf("Max number: %u\n", N);
		printf("%u,", N);

		clock_gettime(CLOCK_REALTIME, &start);
		for(unsigned long i = 0;i < N;++i){
			int count = 0;
			unsigned long num = i;
			while(num > 1){
				if((num & 0x1) == 0) num >>= 1;
				else num = (num << 1) + num + 1;
				count++;

				if(num < i){
					// Stop counting, cause we've already done it
					count += lens[num];
					break;
				}
			}
			lens[i] = count;
			if(lens[i] > lens[max]) max = i;
		}
		clock_gettime(CLOCK_REALTIME, &end);
		timespecsub(&end, &start, &diff);

		//printf("Max chain length is %u, from %lu in %ld.%09lds\n", lens[max], max, diff.tv_sec, diff.tv_nsec);
		printf(" %u, %u, %ld.%09ld\n", lens[max], max, diff.tv_sec, diff.tv_nsec);
	}

	free(lens);
}
