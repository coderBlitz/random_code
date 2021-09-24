#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/random.h>

struct xorshift64_state {
  uint64_t a;
};

uint64_t xorshift64(struct xorshift64_state *state)
{
	uint64_t x = state->a;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	return state->a = x;
}

uint64_t c = 1;
static inline uint64_t lcg64(void){
	return c = 3*c*c + 997*c + 1; // x^3 + x + 1
}

int main(){
	uint64_t state;
	getrandom(&state, sizeof(state), 0);
	srandom(state);
	c = state;

	const uint64_t N = 20;
	const uint64_t M = 7;
	for(long i = 0;i < N;i++){
		printf("%10lu\t", xorshift64((struct xorshift64_state *)&state) % M);
		printf("%10lu\t", random() % M);
		printf("%10lu\n", lcg64() % M);
	}

	return 0;
}
