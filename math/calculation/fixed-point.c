#include<math.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>

// Find root of x^3 + x - 1 = 0
double fn(double x){
	double x2 = x*x;
	double x3 = x2*x;
	return (1 + 2*x3)/(1 + 3*x2);
}

double fn_alt(double x){
	return cbrt(1 - x);
}

double rt2(double x){
	//return x * (1 - x/2) + 1;
	return x * (1 - x/4) + 0.5;
}

double pow_thing(double x){
	return 1.0 + pow(2,x) - pow(1.5,x);
}

// Allow comparing functions more easily
double fpi(double x_init, double (*fn)(double), double tol, uint32_t max_iter){
	uint32_t count = 0;
	double x = x_init;
	double prev; // No need to initialize

	printf("0, %.8lf\n", x_init);
	// Using fixed-point iteration, evaluate function on itself repeatedly
	// Slope near solution must be < 1 for convergence. Absolute value of slope
	// This should be a simplified definition of a Lipschitz function
	do{
		prev = x;
		x = fn(x);

		count++;

		printf("%2u, %.16lf\n", count, x);
	}while(fabs(prev - x) > tol && count < max_iter);
}

int main(int argc, char *argv[]){
	double x_init = 0.5;
	double tol = 0.0000001;
	uint32_t max_iter = 40;

	if(argc == 2){
		printf("Setting initial guess at x = %s\n", argv[1]);
		char *loc = NULL;
		double temp = strtod(argv[1], &loc);
		if(loc == argv[1]){
			fprintf(stderr, "Could not use initial guess. Using default\n");
		}else{
			x_init = temp;
		}
	}

	fpi(x_init, fn, tol, max_iter);

	printf("\n");
	fpi(x_init, fn_alt, tol, max_iter);

	printf("\nRoot 2:\n");
	fpi(1.2, rt2, tol, max_iter);

	printf("\nPower thing\n");
	fpi(-10.0, pow_thing, tol, max_iter);
}
