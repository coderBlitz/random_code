#include<stdio.h>
#include<stdlib.h>

long double factorial(unsigned int i){
	double total = 1;
	for(int n = 2;n <= i;n++) total *= n;
	return total;
}

int main(){
	long double a = 0;
	double b;
	unsigned int n = 0;

	while(n < 30){
		a += 1.0 / factorial(n);

		printf("%.22Lf\n", a);
		n++;
	}
	b = a;
	printf("e = %.18lf\n", b);
}
