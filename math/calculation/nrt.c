#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// Should be faster than pow()
double lpow(double num, long power){
	if(power == 1) return num;
	if(power == 0) return 1.0;

	return num * lpow(num, power-1);
}

// Use newtons method to find n'th root of number to given tolerance
double nrt(long root, double num, double tol){
	uint32_t max_iter = 100; // Can make parameter
	double xn = root + 0.4;
	double osc_check = 0.0;
	double prev;
	double diff = tol + 1;
	long root_minus = root-1;

	for(int i = 0;i < max_iter && fabs(diff) > tol;i++){
		osc_check = prev;
		prev = xn;
		printf("%d\t%.8lf\n", i, xn);
		xn = xn - (lpow(xn, root) - num)/(root_minus*lpow(xn, root_minus));
		if(osc_check == xn){
			printf("Oscillation detected. Stopping.\n");
			break;
		}

		diff = xn - prev;
	}

	return xn;
}

int main(int argc, char *argv[]){
	double num = 123456;
	long root = 7;
	if(argc == 3){
		// Usage: nrt integer_root real_number
		char *check = argv[1];
		double temp = strtod(argv[1], &check);
		if(check != argv[1]) root = temp;
		else printf("Invalid root\n");

		check = argv[2];
		temp = strtod(argv[2], &check);
		if(check != argv[2]) num = temp;
		else printf("Invalid number\n");
	}

	printf("%ld root of %lf is %.10lf\n", root, num, nrt(root, num, 0.000001));
} 
