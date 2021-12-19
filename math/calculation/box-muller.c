/* Generate gaussian distribution using box-muller method
*/

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/random.h>
#include<time.h>

#define RAND64_MAX 0xFFFFFFFFFFFFFFFF

static inline unsigned long random64(void){
	return (random() & 0x7FFF8000) << 33 | (random() & 0x7FFF8000) << 17 | (random() & 0x7FFF8000) << 1 | (random() & 0x7FFF8000) >> 16;
}

int main(int argc, char *argv[]){
	unsigned int seed;
	getrandom(&seed, sizeof(seed), 0);
	srandom(seed);

	long N = 12;
	double sigma = 1.0; // Deviation
	double mu = 0.0; // Mean

	if(argc > 1){
		long tmp = strtoul(argv[1], NULL, 0);
		if(tmp > 0) N = tmp;
	}


	double x1, x2, z1, z2;
	double t1, t2, t3;
	for(long i = 0;i < N;i++){
		x1 = random64() / (double) RAND64_MAX;
		x2 = random64() / (double) RAND64_MAX;
		//printf("(%lf,%lf)\n", x1, x2);

		/* Box-muller calculations
		z1 = sqrt(-2 * ln(x1)) * cos(2 * pi * x2)
		z2 = sqrt(-2 * ln(x1)) * sin(2 * pi * x2)

		Store first part of both equations to reuse (t1)
		Store cosine for next step (t2)
		Use identity sin^2(r) + cos^2(r) = 1 to get sin value (t3)
		*/
		t1 = sqrt(-2.0 * log(x1)); // First part
		t2 = cos(2.0 * M_PI * x2); // Second part (first-eq)
		//t3 = sqrt(1.0 - t2*t2); // sin() calculation
		t3 = sin(2.0 * M_PI * x2);

		z1 = t1 * t2;
		z2 = t1 * t3;

		printf("%lf,%lf\n", z1, z2);
	}
}
