#include<stdio.h>
#include<math.h>

static inline double fn1(double x){
	return -3.0*x*x*x + 5*x -1;
}
static inline double fn1_d(double x){
	return -9.0*x*x + 5;
}

double newtons(double (f)(double), double (F)(double), double xinit, double tol, int max_iter){
	double xn = xinit;
	double err = tol+1;
	while(max_iter-- > 0 && fabs(err) > tol){
		//printf("x1 = %lf\terr=%lf\n", xn, err);
		xn = xn - (f(xn) / F(xn));

		err = f(xn);
	}

	return xn;
}

int main(){
	double tol = 0.0000000001;
	int max_iter = 40;
	double x1 = 1.0;

	double res = newtons(fn1, fn1_d, x1, tol, max_iter);
	printf("Res: %.18lf\n", res);

	printf("Val: %.18lf\n", fn1(res));

	return 0;
}
