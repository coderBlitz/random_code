/***	gcd.c -- Computes GCD of two numbers using extended euclid's algorithm

		Author: Chris Skane
***/

#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char *argv[]){
	uint64_t A = 6643, B = 2873;

	if(argc == 3){
		A = atoll(argv[1]);
		B = atoll(argv[2]);
	}

	if(A == 0 || B == 0){
		printf("Please give non-zero integers\n");
		return 0;
	}

	uint64_t C, D;
	if(A > B){
		D = A;
		C = B;
	}else{
		D = B;
		C = A;
	}

	uint64_t remainder = 0;
	uint64_t quotient = 0;
	do{
		quotient = D/C;
		remainder = D - C*quotient;
		printf("%lu = %lu(%lu) + %lu\n", D, C, quotient, remainder);

		D = C;
		C = remainder;
	}while(remainder != 0);

	printf("GCD(%lu, %lu) = %lu\n", A, B, D);
	if(D == 1){
		printf("Numbers are co-prime\n");
	}
	return 0;
}
