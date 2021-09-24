/*	Gauss-Legendre quadrature
*	Use Newton's method to compute roots of legendre polynomial, using Bonnet's
*	 recursion formula for Legendre polys.
*	Use  weighting function as given by Wikipedia page.
*/

#include<math.h>
#include<stdio.h>
#include<stdlib.h>

/**	Returns N nodes and weights for gaussian quadrature for interval [a,b]
	Bonnet's formula: (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
	Derivative: (x^2-1)/n d/dx P_n(x) = x P_n(x) - P_{n-1}(x) == d/dx P_{n+1}(x) = (n+1) P_n(x) + x d/dx P_n(x)
**/
double leg_quad(const unsigned N, double *X, double *W, double a, double b){
	if(N == 0) return NAN;

	const long double tol = 1.0e-16; // TODO: Make parameter (maybe)

	long double p_n1; // P_{n-1}
	long double p_n; // P_n
	long double ret;

	long double dx_1; // d/dx P_{n+1}
	long double dx; // d/dx P_n

	long double rt; // Root value
	long double rt_p; // Previous root value
	long double x;

	long double mid = (b + a) / 2.0; // Interval midpoint
	long double half_int = (b - a) / 2.0; // Half interval width

	// Loop for half the roots (roots are mirrored). The (N+1)/2 == ceil(N/2), saves a function call
	unsigned ceil_N = (N+1)/2;
	unsigned n;
	for(long i = 1;i <= ceil_N;i++){
		x = cos(M_PI * (4*i - 1)/(long double)(4*N+2)); // Estimator for each root (descending order)
		//printf("x = %lf\n", x);
		rt = x;
		rt_p = rt + tol + 1;

		// Loop for newtons method, using relative error
		while(fabs(rt - rt_p) > tol){
			x = rt;
			dx = 1.0;
			p_n1 = 1.0;
			p_n = x;

			n = 1; // Degree counter
			do{
				dx_1 = (n+1) * p_n + x * dx; // Compute derivative value
				ret = ((2*n + 1) * x * p_n - n * p_n1) / (long double)(n+1); // Compute legendre value

				//printf("n = %3u\tdx = %22.16lf\n", n, dx_1);

				// Iterate stuff
				p_n1 = p_n;
				p_n = ret;
				dx = dx_1;
			}while(++n < N);

			rt_p = rt;
			rt = rt - ret / dx; // Newton iteration
			//printf("rt = %20.16lf\tdx = %20.16lf\n", rt, dx);
		}

		X[i-1] = mid - half_int * rt; // Roots reflected about midpoint
		X[N - i] = mid + half_int * rt;
		W[i-1] = half_int * 2.0 / ((1 - rt*rt) * (dx*dx)); // Weights similarly reflected, albeit always positive
		W[N - i] = W[i-1];
	}

	return tol;
}

int main(int argc, char *argv[]){
	unsigned long N = 5;
	if(argc == 2){
		N = strtoul(argv[1], NULL, 10);
	}

	double rts[N];
	double wts[N];
	leg_quad(N, rts, wts, -1, 1);

	for(long i = 0;i < (N+1)/2;i++){
		printf("[%2d] = ±%20.17lf\t%20.17lf\n", i, fabs(rts[i]), wts[i]);
	}

	return 0;
}
