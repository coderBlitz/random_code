#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double inv( double number ){
		double y = number;
		double x2 = y * 0.5;
		int64_t i = *(int64_t *) &y;
		// The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
		i = 0x5fe6eb50c7b537a9 - (i >> 1);
		y = *(double *) &i;
		y = y * (1.5 - (x2 * y * y));	 // 1st iteration
		y	= y * ( 1.5 - ( x2 * y * y ) );	 // 2nd iteration, this can be removed
		return y*y;
}

// Use newtons method to find inverse to given tolerance
double inverse(unsigned long num, double tol, int max_iter){
	int high = __builtin_clzl(num); // 31/63 for num == 0
	printf("High: %d\n", high);
	unsigned long power = 1 << (64 - high);
	double xn = 1.0/power;

	double osc_check = 0.0;
	double prev;
	double diff = tol + 1;

	for(int i = 0;i < max_iter && fabs(diff) > tol;i++){
		osc_check = prev;
		prev = xn;
		printf("%d\t%.16lf\n", i, xn);
		xn = xn + xn*(1 - num * xn); // More accurate than xn*(2-d*xn)

		diff = xn - prev;
	}

	return xn;
}

int main(int argc, char *argv[]){
	uint32_t max_iter = 100; // Can make parameter
	double tol = 0.000000;
	unsigned long N = 1024;

	if(argc == 2){
		char *temp = argv[1];
		unsigned long val = strtoul(argv[1], &temp, 10);
		if(temp != argv[1]) N = val;
	}

	printf("N = %lu\n", N);
	inverse(N, tol, max_iter);
	printf("%0.16lf\n", inv(N));
}
