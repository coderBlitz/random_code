#include<math.h>
#include<stdio.h>

/*	Trapezoid integration method, that uses Kahan summation algorithm for the sum
*/
double int_trap(double (*fn)(double), double a, double b, unsigned int n){
	if(fn == NULL || n == 0) return NAN;

	double h = (b-a)/n;
	double sum = (fn(a) + fn(b)) / 2.0;
	double c = 0, pt;
	double x, y;
	for(unsigned int i = 1;i < n;++i){
		pt = a + i * h; // Point

		y = fn(pt) - c; // Error correction
		x = sum + y;

		c = (x - sum) - y; // Error accumulator

		sum = x;
	}
	sum *= h;

	return sum;
}

int main(){
	double res;
	for(int n = 16;n <= 128;n *= 2){
		res = int_trap(sin, 0, 1, n);
		printf("n = %d\tRes = %.12lf\n", n, res);
	}
}
