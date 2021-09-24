#include<math.h>
#include<stdio.h>

double id(double x){
	return x;
}
double poly(double x){
	return 1 - x + x*x - x*x*x + x*x*x*x;
}

// Composite simpsons (ish)
double simpson(double (*fn)(double), double a, double b, int N){
	if(N & 0x1) N++;
	double h = (b-a)/N;

	double sum2 = 0;
	double sum4 = 0;

	for(int i = 1;i < N;i++){
		if((i & 0x1)) sum4 += fn(a + i*h);
		else sum2 += fn(a + i*h);
	}

	//printf("Sum2: %.4lf\nSum4: %.4lf\n", sum2, sum4);

	double sum = (fn(a) + 4*sum4 + 2*sum2 + fn(b)) * h/3;
	return sum;
}

int main(){
	double res = 0.0;
	double prev, diff;
	double tol = 0.001;

	for(int i = 2;i <= 100;i += 2){
		prev = res;
		res = simpson(poly, -4, 4, i);
		diff = res - prev;

		printf("%3d\t%.8lf\n", i, res);

		if(fabs(diff) <= tol) break;
	}
}
