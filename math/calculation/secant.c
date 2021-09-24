#include<stdio.h>
#include<math.h>

static inline double fn1(double x){
	return -3.0*x*x*x + 5*x -1;
}

double secant(double (*fn)(double), double x1, double x2, double tol, int max_iter){
	double xn2 = x1;
	double xn = x2;
	double err = tol + 1.0, temp;

	while(max_iter-- > 0 && fabs(err) > tol){
		//printf("x1 = %lf\tx2=%lf\terr=%lf\n", xn, xn2, err);
		temp = xn;
		xn = (xn2 * fn(xn) - xn * fn(xn2)) / (fn(xn) - fn(xn2));
		xn2 = temp;

		err = fn(xn);
	}

	return xn;
}

int main(){
	double tol = 0.0000000001;
	int max_iter = 40;
	double x1 = 1.0, x2 = 1.5;

	double res = secant(fn1, x1, x2, tol, max_iter);
	printf("Res: %.18lf\n", res);

	printf("Val: %.18lf\n", fn1(res));

	return 0;
}
