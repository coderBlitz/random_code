#include<stdio.h>
#include<stdlib.h>

int main(int argc, char *argv[]){
	unsigned long N = 1;
	if(argc == 2){
		N = strtoul(argv[1], NULL, 10);
	}else{
		printf("Usage: collatz NUM");
		return 1;
	}

	printf("%lu\n", N);
	while(N > 1){
		if(N % 2 == 0) N /= 2;
		else N = N*3 + 1;

		printf("%lu\n", N);
	}

	return 0;
}
