#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

/***	General notes
	- With seed 10, first number to repeat is 914001081 with indices {57039, 75053}
	- Most starting seeds yield cycle length of 18014
	- Seed 1271565896 has cycle length 211, starting at 308
***/

uint32_t myRand(uint32_t seed){
	srand(seed);
	return rand();
}

uint32_t test_lcg(uint32_t seed){
	return (103 * seed) % 131;
}

uint32_t mimic_lcg(uint32_t seed){
	return (103 * seed) + 1804289383 % RAND_MAX;
}


int main(int argc, char *argv[]){
	uint64_t N = (uint64_t)RAND_MAX+1;
	uint32_t base_seed = 0;
	const uint32_t num_seeds = 50;
	int seed;

	if(argc == 2){
		int temp = atol(argv[1]);
		if(temp > 0) base_seed = temp;
	}
	//printf("N: %lu\n", N);
	//uint32_t (*fn)(uint32_t) = myRand;
	uint32_t (*fn)(uint32_t) = test_lcg;

	uint32_t last_seed = base_seed + num_seeds;
	printf("seed, cycle length, start\n");
	for(int j = base_seed;j <= last_seed;j++){
		/*srand(j);
		seed = rand();*/
		seed = j;
		//printf("%d,", seed);
		uint8_t *num = calloc((uint64_t)RAND_MAX+1, sizeof(*num));
		if(num == NULL){
			fprintf(stderr, "Could not alloc array.\n");
			return -1;
		}

		/* Find lambda and mu such that x_{mu} = x_{lambda + mu} (the cycle)
		First find lambda (sequence length), then move hare to lambda, then move both to find mu
		*/
		uint32_t p = 1, lam = 1;
		uint32_t t = seed;
		uint32_t h = fn(t);
		while(t != h){
			if(lam == p){
				p *= 2; // Increment p in any way (exponential best)
				lam = 0;
				t = h;
			}

			h = fn(h);
			lam += 1;
		}

		if(lam == N) printf("No cycle found within %lu iterations\n", N);

		// Separate hare and turtle by lambda
		t = seed;
		h = t;
		for(long i = 0;i < lam;++i){
			h = fn(h);
		}

		// Find mu (start) of sequence
		uint32_t mu = 0;
		while(t != h){
			t = fn(t);
			h = fn(h);
			++mu;
		}

		// Output
		printf("%lu,%lu,%lu\n", seed, lam, mu);

		free(num);
	}

	return 0;
}
