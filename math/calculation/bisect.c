#include<stdio.h>

double fn1(double x){
	return x * x - 2;
}

double bisect(double (*fn)(double), double a, double b, double tol){
	double c, d, err = tol+1.0;

	// Fix f(a) < 0, f(b) > 0
	if(fn(a) > 0){
		double t1 = a;
		a = b;
		b = t1;
	}

	// Do bisection
	while(err > tol){
		c = (a + b) / 2.0;

		err = c - a; // err is now (b-a)/2

		if(fn(c) < 0){
			a = c;
		}else{
			b = c;
		}
	}

	return (a+b) / 2.0;
}

int main(){
	double tol = 0.0000000001;
	double a = 0.0;
	double b = 2.0;

	double (*fn)(double) = fn1;

	double res = bisect(fn, a, b, tol);
	printf("Res: %.18lf\n", res);

	printf("Val: %.18lf\n", fn(res));

	return 0;
}
