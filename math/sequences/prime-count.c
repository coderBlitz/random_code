#include<stdio.h>
#include<stdlib.h>

int main(int argc, char *argv[]){
	size_t N = 50;
	if(argc == 2){
		char *tmp;
		size_t temp = strtoull(argv[1], &tmp, 10);
		if(tmp != argv[1]) N = temp;
	}

	printf("N: %lu\n", N);

	if(N == 2 || N == 3){
		printf("%d\n", N-1);
		return 0;
	}

	size_t count = 2;
	char flag;
	for(size_t i = 5;i <= N;i += 6){
		flag = 1;
		for(size_t j = 3; j*j < i; j += 2){
			if(i % j == 0){
				flag = 0;
				break;
			}
		}
		if (flag) count++;

		i += 2;
		if(i > N) break;
		flag = 1;
		for(size_t j = 3; j*j < i; j += 2){
			if(i % j == 0){
				flag = 0;
				break;
			}
		}
		if (flag) count++;
	}

	printf("%d\n", count);
	return 0;
}
